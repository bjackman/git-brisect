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

    def write_executable(self, path, content):
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o777)
        self.git("add", path)

    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def test_no_tests_needed(self):
        # TODO: Actually we should check that the script doesn't get run.
        self.write_executable("run.sh", "true")
        good = self.commit()
        self.write_executable("run.sh", "exit 1")
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("bisect", "bad", bad)

        self.assertEqual(git_dissect.dissect(["run.sh"]), bad, f"good: {good} bad: {bad}")

    def test_smoke(self):
        # Linear history where the code gets broken in the middle.
        self.write_executable("run.sh", "true")
        good = self.commit()
        self.commit()
        self.write_executable("run.sh", "exit 1")
        want = self.commit()
        bad = self.commit()

        self.git("bisect", "start")
        self.git("bisect", "good", good)
        self.git("biset", "bad", bad)

        self.logger.info(self.git("log", "--graph", "--all", "--format=%H %d"))

        self.assertEqual(git_dissect.dissect(["sh", "./run.sh"]), want)

    def test_nonlinear_multiple_good(self):
        # Linear history where the code is good in both branches and then broken after a merge
        self.write_executable("run.sh", "true")
        base = self.commit()
        good1 = self.commit()
        self.git("checkout", base)
        good2 = self.commit()
        self.git("merge", good2)
        self.write_executable("run.sh", "exit 1")
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)

    unittest.main()