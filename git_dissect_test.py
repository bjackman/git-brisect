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

        self.git("init")

    def git(self, *args):
        res = subprocess.run(("git",) + args, capture_output=True)
        try:
            res.check_returncode()
        except:
            print(f"Git command {args} failed.")
            print(res.stderr.decode())
            raise
        return res.stdout.decode().strip()

    def commit(self, msg="dummy commit message"):
        self.git("commit", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def write_script(self, path, fail):
        """Write a script to path that can be used to "test" a commit.

        If fail, the script will exit with an error code.

        The script writes to a file so that script_runs can detect when it
        gets run.

        This also git adds the script.
        """
        content = f"git rev-parse HEAD >> {os.getcwd()}/output.txt"
        if fail:
            content += "; exit 1"

        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o777)

        self.git("add", path)

    def script_runs(self, path):
        """Return each of the commits tested by the script from write_script."""
        if not os.path.exists(path):
            return []
        with open("output.txt") as f:
            return [l.strip() for l in f.readlines()]

    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def test_no_tests_needed(self):
        # TODO: Actually we should check that the script doesn't get run.
        self.write_script("run.sh", fail=False)
        good = self.commit()
        self.write_script("run.sh", fail=True)
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)

        self.assertEqual(git_dissect.dissect(["run.sh"]), bad)
        self.assertFalse(os.path.exists("output.txt"), "Script was run unnecessarily")

    def test_smoke(self):
        # Linear history where the code gets broken in the middle.
        self.write_script("run.sh", fail=False)
        good = self.commit()
        want_test1 = self.commit()
        self.write_script("run.sh", fail=True)
        want_culprit = self.commit()
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("biset", "bad", bad)

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want_culprit)
        self.assertCountEqual(
            self.script_runs("run.sh"),
            [want_test1, want_culprit],
            "didn't get the expected set of script runs")

    def test_nonlinear_multiple_good(self):
        # Linear history where the code is good in both branches and then broken after a merge
        self.write_script("run.sh", fail=False)
        base = self.commit()
        good1 = self.commit()
        self.git("checkout", base)
        good2 = self.commit()
        self.git("merge", good2)
        self.write_script("run.sh", fail=True)
        want = self.commit()
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good1)
        self.git("bisect", "good", good2)
        self.git("bisect", "bad", bad)

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want)


    # TODO: above case but it gets broke in one of the branches

    # TODO:
    #  nonlinear, multiple good
    #  nonlinear, single good
    #  worktree mode (check run in expected dir)
    #  non-worktree mode
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