#!/usr/bin/python3

import logging
import unittest
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading

import git_dissect

class GitDissectTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(self.id())

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir))
        os.chdir(tmpdir)

        self.commit_counter = 0

        self.git("init")

    # Why does the test have a different style git wrapper function than the
    # main code? It just does, OK! Stop asking annoying questions!
    def git(self, *args):
        res = subprocess.run(("git",) + args, capture_output=True)
        try:
            res.check_returncode()
        except:
            print(f"Git command {args} failed.")
            print(res.stderr.decode())
            raise
        return res.stdout.decode().strip()

    def commit(self, msg=None):
        if msg is None:
            msg = f"commit {self.commit_counter}"
            self.commit_counter += 1
        self.git("commit", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def write_executable(self, path, content):
        """Write an executable and git add it"""
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o777) # SIR yes SIR o7

        self.git("add", path)

    def write_pass_fail_script(self, fail):
        """Write a script to path that can be used to "test" a commit.

        If fail, the script will exit with an error code.

        The script writes to a file in the current CWD (not the CWD the script
        runs in) that script_runs can use to detect when it gets run.

        This also git adds the script.
        """
        content = f"git rev-parse HEAD >> {os.getcwd()}/output.txt"
        if fail:
            content += "; exit 1"
        self.write_executable("run.sh", content)

    def read_stripped_lines(self, path):
        with open(path) as f:
            return [l.strip() for l in f.readlines()]

    def script_runs(self):
        """Return each of the commits tested by the script from write_script."""
        if not os.path.exists("output.txt"):
            return []
        return self.read_stripped_lines("output.txt")

class TestBisection(GitDissectTest):
    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def setup_bisect(self, good, bad):
        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)

    def test_no_tests_needed(self):
        # TODO: Actually we should check that the script doesn't get run.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        self.write_pass_fail_script(fail=True)
        bad = self.commit()

        self.setup_bisect(good, bad)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["sh","run.sh"]), bad)
        self.assertFalse(os.path.exists("output.txt"), "Script was run unnecessarily")

    def test_smoke(self):
        # Linear history where the code gets broken in the middle.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        want_test1 = self.commit()
        self.write_pass_fail_script(fail=True)
        want_culprit = self.commit()
        bad = self.commit()

        self.setup_bisect(good, bad)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want_culprit)
        self.assertCountEqual(self.script_runs(),
                              [want_test1, want_culprit],
                              "didn't get the expected set of script runs")

    def test_nonlinear_multiple_good(self):
        # Branched history where the code is good in both branches and then broken after a merge
        self.write_pass_fail_script(fail=False)
        base = self.commit()
        good1 = self.commit()
        self.git("checkout", base)
        good2 = self.commit()
        self.git("merge", "--no-edit", good1)
        merge = self.git("rev-parse", "HEAD")
        self.write_pass_fail_script(fail=True)
        want = self.commit()
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good1)
        self.git("bisect", "good", good2)
        self.git("bisect", "bad", bad)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want)
        self.assertCountEqual(self.script_runs(), [merge, want],
                              "didn't get expected set of script runs")

    def test_bug_in_branch(self):
        # Branched history where the bug arises in one of the branches.
        self.write_pass_fail_script(fail=False)
        good = self.commit("good")
        other = self.commit("other")
        self.git("checkout", good)
        self.write_pass_fail_script(fail=True)
        want = self.commit("want")
        also_test = self.commit("also_test")
        self.git("checkout", other)
        self.git("merge", "--no-edit", also_test)
        bad = self.commit("bad")

        self.setup_bisect(good, bad)
        self.logger.info("\n" + self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want)
        # All we do is find the first commit in the bisection range where things
        # went from good to bad (this is just how git-bisect works). It's
        # reasonable that we only tested exactly the two commits we needed to
        # achieve that.
        runs = self.script_runs()
        self.assertIn(want, runs)

    def _run_worktree_test(self, use_worktrees: bool, cleanup_worktrees=False):
        """Run a test that should result in multiple worktrees being used.

        (Unless use_worktrees is false). The test script records what CWD it was
        run in, and leaves a file behind. It also records whether it found that
        file left over in the CWD from a previous run. Returns a list of pairs
        of (CWD, dirty), where "dirty" actually just means that the specific
        file was left behind.

        Does its best to ensure that at least one worktree gets reused, by
        having much more commits than threads. In theory this might actually
        flake though, if by chance the bisection algorithm lands on testing the
        exact 2 commits on either side of the breakage before any others.
        """
        script = f"""
            [ -e ./dirty ]; dirty_status=$?
            echo $(pwd) "$dirty_status" >> {os.getcwd()}/cwds.txt
            touch ./dirty
        """
        self.write_executable("log_cwd.sh", script)

        self.write_pass_fail_script(fail=False)
        init = self.commit("init")
        for i in range(50):
            self.commit("good " + str(i))
        self.write_pass_fail_script(fail=True)
        end = self.commit("end")

        self.setup_bisect(init, end)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        git_dissect.dissect(["sh", "./log_cwd.sh", "./run.sh"],
                            use_worktrees=use_worktrees, cleanup_worktrees=cleanup_worktrees,
                            num_threads=4)
        runs = self.script_runs()

        lines = self.read_stripped_lines("cwds.txt")
        # It's possible in theory that all but one of the threads was very slow,
        # and a single thread ended up doing all the work. And maybe by chance
        # it happened to test exactly the commits it needed, so we might just
        # get the same cwd twice. Seems unlikely so we don't bother to try and
        # avoid any case like this even though it might reduce test coverage.
        self.assertGreaterEqual(len(lines), 2)
        ret = []
        for line in lines:
            cwd, dirty_status = line.split(" ")
            self.assertIn(dirty_status, ("0", "1"))
            ret.append((cwd, dirty_status == "0"))
        return ret

    def test_worktree_mode(self):
        for (cwd, dirty) in self._run_worktree_test(use_worktrees=True,
                                                    cleanup_worktrees=True):
            self.assertNotEqual(cwd, os.getcwd())
            # This is not actually required, any directory is fine, just a hack
            # assertion to try and catch scripts getting run in totally random
            # trees.
            self.assertNotEqual(cwd.find("worktree-"), -1)
            self.assertFalse(dirty)

    def test_worktree_mode_no_cleanup(self):
        results = self._run_worktree_test(use_worktrees=True,
                                          cleanup_worktrees=False)
        # If this flakes, it might be that by chance no worktree got reused,
        # which isn't technically a bug. Try to find a way to force them to get
        # reused.
        self.assertTrue(any(dirty for (cwd, dirty) in results))

    def test_non_worktree_mode(self):
        results = self._run_worktree_test(use_worktrees=False)

        # If nobody saw a dirty tree, seems like there was more parallelism than
        # expected, or we were racily cleaning up the tree.
        self.assertTrue(any(dirty for (cwd, dirty) in results))
        self.assertTrue(all(cwd == os.getcwd() for (cwd, dirty) in results))

    def _test_thread_limit(self, num_threads):
        # At each commit will be a script that starts, touches a unique file
        # path, then hangs until a different unique filepath appers, then exits.
        commits = []
        for i in range(20):
            script = f"""
                touch {os.getcwd()}/started-{i}
                while ! [ -e {os.getcwd()}/stop-{i} ]; do
                    sleep 0.01
                done
                touch {os.getcwd()}/done-{i}
                exit {"1" if i > 17 else "0"}
            """
            self.write_executable("run.sh", script)
            commits.append(self.commit())

        # Run the dissection in the background
        self.setup_bisect(commits[0], commits[-1])
        dissect_result = None
        def run_dissect():
            nonlocal dissect_result
            dissect_result = git_dissect.dissect(args=["sh", "./run.sh"],
                                         num_threads=num_threads)

        thread = threading.Thread(target=run_dissect)
        thread.start()

        # Gets the indexes of the commits whose test scripts are currently
        # started/done
        def tests_in_state(state):
            ret = set()
            for i in range(len(commits)):
                if os.path.exists(state + "-" + str(i)):
                    ret.add(i)
            return ret
        def running_tests():
            return tests_in_state("started") - tests_in_state("done")

        # Free up one thread at a time. We will do this at most once for each
        # remaining commit (i.e. excluding the ones whose tests are already
        # running)
        for i in range(len(commits) - num_threads):
            # No choice really but to just sleep for some random time, we can
            # detect when the expected tests get started, but we don't know when
            # we can trust that it doesn't also start any unexpected tests.
            time.sleep(0.5)

            if not thread.is_alive():
                break

            # TODO: I think the failure here is a race condition in the thread
            # pool code. Probably instead of debugging, best to just clean it
            # up.
            self.assertEqual(len(tests_in_state("started")), i + num_threads,
                             "after stopping %d threads" % i)

            # Let one of the threads exit
            to_terminate = next(iter(running_tests()))
            open("stop-" + str(to_terminate), "w").close()

        # If this fails then dissect() must have crashed.
        self.assertIsNotNone(dissect_result)

        thread.join()

    def test_no_parallelism(self):
        self._test_thread_limit(1)

    def test_limited_threads(self):
        self._test_thread_limit(4)

    def test_many_threads(self):
        self._test_thread_limit(4)

    def _test_test_selection(self, num_good, num_bad, want_num_tests):
        # This runs a test to check that we're actually bisecting and not just
        # testing every commit or picking random commits or something.
        self.write_pass_fail_script(fail=False)
        commits = []
        for _ in range(num_good):
            commits.append(self.commit())
        self.write_pass_fail_script(fail=True)
        for _ in range(num_bad):
            commits.append(self.commit())

        self.setup_bisect(commits[0], commits[-1])
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))
        result = git_dissect.dissect(["sh", "./run.sh"], num_threads=1)

        # Check we didn't cheat
        tested_commits = self.script_runs()
        self.assertEqual(result, commits[num_good])
        self.assertIn(commits[num_good - 1], tested_commits)
        self.assertIn(commits[num_good], tested_commits)

        self.assertEqual(len(tested_commits), want_num_tests)

    def test_test_selection_1(self):
        self._test_test_selection(10, 5, 4)

    def test_test_selection_2(self):
        self._test_test_selection(5, 10, 4)

    def test_test_selection_3(self):
        self._test_test_selection(1, 1, 2)

    # TODO:
    #
    #  pathological args? Like good and bad aren't ancestors?
    #
    #  ensuring script isn't run more times than necessary? Maybe done
    #
    #  ensuring test cleanups happen
    #
    #  gathering output
    #
    #  cancelation of tests that turn out to be unnecessary
    #
    #  ensure things don't break under long enqueuements

class TestTestEveryCommit(GitDissectTest):
    def test_smoke(self):
        self.write_pass_fail_script(fail=False)
        commits = []
        commits.append(self.commit())
        commits.append(self.commit())
        commits.append(self.commit())
        self.write_pass_fail_script(fail=True)
        commits.append(self.commit())
        commits.append(self.commit())
        commits.append(self.commit())
        self.write_pass_fail_script(fail=False)
        commits.append(self.commit())

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        results = git_dissect.test_every_commit(
            ["sh", "./run.sh"],
            exclude=[commits[0]], include=[commits[-1]])
        want = list(reversed(list(zip(commits[1:], [
            0, 0, 1, 1, 1, 0
        ]))))

        self.assertEqual(results, want)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)-6s %(message)s")

    unittest.main()