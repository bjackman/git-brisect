#!/usr/bin/python3

import unittest
import os
import shutil
import subprocess
import tempfile

import git_dissect

class TestGitDissect(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir))
        os.chdir(tmpdir)

        self.git(["init"])

    def git(self, args):
        res = subprocess.run(["git"] + args, capture_output=True)
        res.check_returncode()
        return res.stdout

    def commit(self, msg="dummy commit message"):
        self.git(["commit", "-m", msg])
        return self.git(["rev-list", "HEAD"])

    def write(self, path, content):
        with open(path, "w") as f:
            f.write(content)
        self.git(["add", path])

    def test_not_bisect(self):
        with self.assertRaises(git_dissect.NotBisectingError):
            git_dissect.dissect([])

    def test_smoke(self):
        self.write("run.sh", "true")
        good = self.commit()
        self.write("run.sh", "exit 1")
        bad = self.commit()

        self.git(["bisect", "start"])

if __name__ == "__main__":
    unittest.main()