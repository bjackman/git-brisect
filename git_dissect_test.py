#!/usr/bin/python3

import unittest
import os
import shutil
import subprocess
import sys
import tempfile

import git_dissect

class TestGitDissect(unittest.TestCase):
    def setUp(self):
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
        self.git("commit", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def write(self, path, content):
        with open(path, "w") as f:
            f.write(content)
        self.git("add", path)

    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def test_no_tests_needed(self):
        self.write("run.sh", "true")
        good = self.commit()
        self.write("run.sh", "exit 1")
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("biset", "bad", bad)

        self.assertEqual(git_dissect.dissect(["run.sh"]), bad, f"good: {good} bad: {bad}")

    # TODO:
    #  linear case
    #  nonlinear, multiple good
    #  nonlinear, single good
    #  worktree mode
    #  non-worktree mode
    #  replacing args


if __name__ == "__main__":
    unittest.main()