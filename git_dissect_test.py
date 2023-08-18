#!/usr/bin/python3

import logging
import unittest
import os
import shutil
import subprocess
import sys
import tempfile

import git_dissect

class TestGitDissect(unittest.TestCase):
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

    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def test_no_tests_needed(self):
        # TODO: Actually we should check that the script doesn't get run.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        self.write_pass_fail_script(fail=True)
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["run.sh"]), bad)
        self.assertFalse(os.path.exists("output.txt"), "Script was run unnecessarily")

    def test_smoke(self):
        # Linear history where the code gets broken in the middle.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        want_test1 = self.commit()
        self.write_pass_fail_script(fail=True)
        want_culprit = self.commit()
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)
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
        good = self.commit()
        optional = self.commit()
        self.git("checkout", good)
        self.write_pass_fail_script(fail=True)
        want = self.commit()
        also_test = self.commit()
        test_me = [also_test]
        self.git("checkout", optional)
        self.git("merge", "--no-edit", also_test)
        test_me.append(self.git("rev-parse", "HEAD"))
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)
        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want)
        # All we do is find the first commit in the bisection range where things
        # went from good to bad (this is just how git-bisect works). So if we
        # find that commit before testing the commit in the other branch, we can
        # abort testing the other branch. Therefore it's optional.
        runs = self.script_runs()
        if optional in runs:
            runs.remove(optional)
        self.assertCountEqual(runs, [want] + test_me,
                              "didn't get expected set of script runs")

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

        self.git("bisect", "start")
        self.git("bisect", "good", init)
        self.git("bisect", "bad", end)
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



    # TODO:
    #  test num_threads
    #  replacing args
    #  ensuring script isn't run more times than necessary
    #  ensuring test cleanups happen
    #  gathering output
    #  bisect should get reset afterwardsj
    #  cancelation of tests that turn out to be unnecessary
    #  ensure things don't break under long enqueuements


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)

    unittest.main()