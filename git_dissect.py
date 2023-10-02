#!/usr/bin/python3
from __future__ import annotations

"""
Usage:
git bisect start
git bisect bad $BAD
git bisect good $GOOD
git-dissect [--[no-]worktrees] $*
"""

import argparse
import dataclasses
import logging
import os
import signal
import subprocess
import tempfile
import threading
import typing
import queue
import shutil
import sys

from typing import Optional
from collections.abc import Iterable

logger = logging.getLogger(__name__)

class CalledProcessError(Exception):
    pass

def run_cmd(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True)
    # subprocess.CompletedProcess.check_returncode doesn't set stderr.
    if result.returncode != 0:
        raise CalledProcessError(
            f'Command {args} failed with code {result.returncode}. ' +
            f'Stderr:\n{result.stderr.decode()}\nStdout:\n{result.stdout.decode()}')
    return result.stdout.decode()

def rev_parse(rev):
    return run_cmd(["git", "rev-parse", rev]).strip()

def rev_list(spec):
    return run_cmd(["git", "rev-list", spec]).splitlines()

def list_parents(rev):
    results = run_cmd(["git", "rev-list", "--parents", "-n", "1", rev]).strip()
    return results.split(" ")[1:]  # First item is @rev.

def merge_base(*commits):
    if len(commits) == 1:
        return commits
    return run_cmd(["git", "merge-base"] + list(commits)).strip()

def describe(rev):
    return run_cmd(["git", "describe", "--tags", "--always", rev]).strip()

class BadRangeError(Exception):
    pass

class RevRange:
    # Git bisect always has one "include" i.e. the single good ref. I don't
    # really know why there is 1 good ref but N bad ones. But we just follow
    # git's lead. I thik this would probably be obvious if you tried to
    # implement this algorithm in a way where both ends of the range can be
    # plural?
    def __init__(self, exclude: list[str], include: str):
        self.exclude = list(set(exclude))
        self.include = include

        self._commits: Optional[set[str]] = None
        self._midpoint: Optional[str] = None

    @classmethod
    def from_string(cls, s):
        # TODO test
        exclude = []
        include = []

        # Would generally have multiple tips. If the user can't specify it with
        # '..' then we can't dissect it.
        if "..." in s:
            raise BadRangeError("Can't dissect ranges specified with '...'. Did you mean '..'?")

        for part in s.split(" "):
            if not part:
                continue
            elif ".." in part:
                dd_parts = part.split("..")
                if len(dd_parts) != 2:
                    raise BadRangeError("%s contains more than one '..'" % s)
                exclude.append(dd_parts[0])
                include.append(dd_parts[1])
            elif part.startswith("^"):
                exclude.append(part[1:])
            else:
                include.append(part)
        if len(include) == 0:
            raise BadRangeError(f"No commits included in range {s}")
        if len(include) > 1:
            # I don't realy know why this restriction exists; it would
            # complicate the implementation in some nontrivial ways but I
            # feel it could be done. Would be interesting to find out why
            # the normal git-bisect doesn't allow it.
            raise BadRangeError(f"Can't dissect with multiple tip revs: {' '.join(include)}")
        return cls(exclude=exclude, include=include[0])

    def __str__(self):
        return "RevRange([%s %s] %d commits)" % (
            describe(self.include), " ".join("^" + describe(e) for e in self.exclude), len(self.commits()))

    def _get_commits(self):
        if self._commits is not None:
            return

        # Find commits in range, sorted by descending distance to nearest
        # endpoint.
        args =  (["git", "rev-list", "--bisect-all"] +
                 [self.include] + ["^" + r for r in self.exclude])
        out = run_cmd(args)
        commits_by_distance = [l.split(" ")[0] for l in out.splitlines()]

        if len(commits_by_distance):
            self._midpoint = commits_by_distance[0]
        self._commits = set(commits_by_distance)

    def midpoint(self) -> Optional[str]:
        self._get_commits()
        return self._midpoint

    def commits(self) -> set[str]:
        self._get_commits()
        # Type checker can't tell this is no longer optional.
        return typing.cast(set[str], self._commits)

    def split(self, commit: str) -> tuple[RevRange, RevRange]:
        """Split into two disjoint subranges at the given commit.

        Produces a "before" subrange that is the input commit and all of its
        ancestors within this range, and an "after" subrange that is the
        complement of that.
        """
        if commit not in self.commits():
            # Actually we could just return self and an empty range. But since
            # we don't expect to need this, we just fail at the moment.
            raise RuntimeError(f"{commit} not in {self}")

        before = RevRange(exclude=self.exclude, include=commit)
        # To ensure the two subranges are non-overlapping, exclude their common
        # ancestor. ACTUALLY, is there a bug here? We generate two ranges with
        # no gap between them; normally a gap between ranges represents some
        # knowledge that we have, but not here.
        mb = merge_base(self.include, commit)
        after = RevRange(exclude=self.exclude + [commit, mb], include=self.include)
        return before, after

    def drop_include(self) -> list[RevRange]:
        """Return disjoint subranges of this range, without @include

        Returns one subrange for each of @include's parents. These subranges are
        a partition of this range, minus its @include.
        """
        # Note - If we have an efficient implementation of this algorithm, I
        # think we can drop the "one include" restriction.
        parents = list_parents(self.include)
        ret = []

        # @parents generally have overlapping ancestors - do some merge-base
        # trickery to get disjoint subranges. I had a hard time coming up with
        # this, if it looks wrong it might be. Basically this felt like it might
        # work and when I tried it out on paper it worked. Then instead of
        # managing to really underestand it I just wrote that whole
        # galaxy-brained Hypothesis-based test suite, really just to see if this
        # worked. So, only trust this code to the extent you trust the tests.
        for i, parent in enumerate(parents):
            prior_parents = [parents[p] for p in range(i)]
            merge_bases = [merge_base(parent, prior) for prior in prior_parents]
            ret.append(RevRange(exclude=self.exclude + merge_bases, include=parent))
        return ret

class WorkerPool:
    # Note that there really shouldn't be any need for multi-threading here, we
    # ought to just have a single thread and a bunch of subprocesses that
    # directly run the user's test script. The problem is Python doesn't provide
    # a portable way to select on a set of concurrent subproccesses. Also, why
    # this condition variable quadrille for the inter-thread communication,
    # can't we just do this like we would in Go? Yeah I dunno. queue.Queue
    # doesn't seem to have an equivalent to closing a Go channel so you need
    # some out-of-band "done" signal and at that point it seems cleaner to just
    # roll a custom mechanism from scratch.

    # TODO: create worktrees lazily -> support unlimited parallelism.

    def __init__(self, test_cmd, workdirs, cleanup_worktrees):
        # Used to synchronize enqueuement with the worker threads.
        self._cond = threading.Condition()
        self._in_q = []
        self._out_q = []
        self._threads = []
        self._subprocesses = {}
        self._done = False
        self._test_cmd = test_cmd
        self._cleanup_worktrees = cleanup_worktrees

        for workdir in workdirs:
            t = threading.Thread(target=self._work, args=(workdir,))
            t.start()
            self._threads.append(t)
        self.num_threads = len(self._threads)

    def enqueue(self, rev):
        with self._cond:
            self._in_q.append(rev)
            logger.info(f"Enqueued {describe(rev)}, new queue depth {len(self._in_q)}")
            # TODO: Because we use the same condition variable for input and
            # output, we need notify_all. Rework this to avoid that.
            self._cond.notify_all()

    def num_pending(self):
        with self._cond:
            return len(self._in_q) + len(self._subprocesses)

    def out_q_length(self):
        with self._cond:
            return len(self._out_q)

    def cancel(self, range: RevRange):
        with self._cond:
            for rev, p in self._subprocesses.items():
                if rev in range.commits():
                    # TODO: Does this work on Windows?
                    self._subprocesses[rev].send_signal(signal.SIGINT)

    def _work(self, workdir):
        while True:
            with self._cond:
                while not self._in_q and not self._done:
                    self._cond.wait()
                if self._done:
                    return

                rev = self._in_q[0]
                self._in_q = self._in_q[1:]

                # TODO: Capture stdout and stderr somewhere useful.
                run_cmd(["git", "-C", workdir, "checkout", rev])
                if self._cleanup_worktrees:
                    run_cmd(["git", "-C", workdir, "clean", "-fdx"])
                try:
                    p = subprocess.Popen(
                        self._test_cmd, cwd=workdir, shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except (PermissionError, FileNotFoundError) as e:
                    logging.info(f"Error running at {rev} ({e}), returning code 1")
                    self._out_q.append((rev, 1))
                    self._cond.notify_all()
                    continue

                self._subprocesses[rev] = p

            p.communicate()
            with self._cond:
                self._out_q.append((rev, p.returncode))
                del self._subprocesses[rev]
                self._cond.notify_all()

    def wait(self):
        with self._cond:
            while len(self._out_q) == 0:
                self._cond.wait()
            ret = self._out_q[0]
            self._out_q = self._out_q[1:]
            return ret

    def interrupt_and_join(self):
        with self._cond:
            for p in self._subprocesses.values():
                p.send_signal(signal.SIGINT)
            self._done = True
            self._cond.notify_all()
        for t in self._threads:
            t.join()

def do_dissect(args, pool, full_range):
    # Non-overlapping ranges of commits that we haven't yet kicked off tests
    # for, and which haven't been excluded from full_range.
    subranges = [full_range]
    while len(full_range.commits()):
        # Start as many worker threads as possible, unless there's a result
        # pending; that will influence which commits we need to test so there's
        # po point in adding new ones until we've processed it. Here we do a
        # breadth-first bisection of the overall range to divide it among
        # available threads. That's not a fully optimal distribution; we want
        # some algorithm that minimizes the average distance between the commits
        # we test. But I can't think of such an algorithm; this one is easy to
        # implement, still kinda fun, and produces not completely insane
        # behaviour.
        while subranges and pool.num_pending() < pool.num_threads and not pool.out_q_length():
            r = subranges.pop(0)
            if not r.commits():
                continue

            # Start testing the midpoint, then split the r into subranges,
            # we'll process them later.
            pool.enqueue(r.midpoint())
            subranges += r.split(r.midpoint())

            # The commit we'll learn most by testing is the midpoint of the
            # largest remaining subrange. Sorting by size makes this a sort of
            # hill-climbing search. subranges = sorted(subranges, key=lambda r:
            subranges = sorted(subranges, key=lambda r: len(r.commits()), reverse=True)

        result_commit, returncode = pool.wait()
        if returncode == 0:  # Commits before are now known-good.
            (cancel, full_range) = full_range.split(result_commit)
        else:                # Commits after are now known-bad.
            (full_range, cancel) = full_range.split(result_commit)
        logger.info(f"Got result {returncode} for {describe(result_commit)}")
        logger.info(f"    Canceling {cancel}, remaining: {full_range}")

        pool.cancel(cancel)
        # Abort ongoing tests to free up threads.
        # Drop known-* commits from future testing. subranges are either subsets
        # of cancel, disjoint from it. (If one wasn't, that must imply we kicked
        # off a test for a commit within it, in which case we should have split
        # it up).
        subranges = [s for s in subranges if s.midpoint() in full_range.commits()]

    return rev_parse(full_range.include)

def excepthook(*args, **kwargs):
    threading.__excepthook__(*args, **kwargs)  # pytype: disable=module-attr
    # Not sure exactly why sys.exit doesn't work here. This is cargo-culted from:
    # https://github.com/rfjakob/unhandled_exit/blob/e0d863a33469/unhandled_exit/__init__.py#L13
    os._exit(1)

def dissect(rev_range: str, args: Iterable[str], num_threads=8, use_worktrees=True, cleanup_worktrees=False):
    tmpdir = None
    if use_worktrees:
        tmpdir = tempfile.mkdtemp()
        worktrees = [os.path.join(tmpdir, f"worktree-{i}") for i in range(num_threads)]
    else:
        worktrees = []

    pool = None
    try:
        for w in worktrees:
            run_cmd(["git", "worktree", "add", w, "HEAD"])
        pool = WorkerPool(args,
                          worktrees or [os.getcwd() for _ in range(num_threads)],
                          cleanup_worktrees=cleanup_worktrees)
        return do_dissect(args, pool, RevRange.from_string(rev_range))
    finally:
        if pool:
            pool.interrupt_and_join()
        for w in worktrees:
            run_cmd(["git", "worktree", "remove", "--force", w])
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

def do_test_every_commit(pool, commits):
    for commit in commits:
        pool.enqueue(commit)

    results = []
    for _ in range(len(commits)):
        results.append(pool.wait())

    return results

# include and exclude specify the set of commits to test.
def test_every_commit(rev_range: str, args: Iterable[str],
                      num_threads=8, use_worktrees=True, cleanup_worktrees=False):
    commits = RevRange.from_string(rev_range).commits()
    tmpdir = None
    if use_worktrees:
        tmpdir = tempfile.mkdtemp()
        worktrees = [os.path.join(tmpdir, f"worktree-{i}") for i in range(num_threads)]
    else:
        worktrees = []

    pool = None
    try:
        for w in worktrees:
            run_cmd(["git", "worktree", "add", w, "HEAD"])
        pool = WorkerPool(args,
                          worktrees or [os.getcwd() for _ in range(num_threads)],
                          cleanup_worktrees=cleanup_worktrees)
        return do_test_every_commit(pool, commits)
    finally:
        if pool:
            pool.interrupt_and_join()
        for w in worktrees:
            run_cmd(["git", "worktree", "remove", "--force", w])
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

def parse_args():
    parser = argparse.ArgumentParser(
        description="git bisect and rebase --exec, but with parallelism")
    # TODO: add short args (not sure how to do this, no docs on the plane!)
    def positive_int(val):
        i = int(val)
        if i < 0:
            raise ValueError("value cannot be negative")
        return i
    # TODO: add short options (-n etc)
    parser.add_argument(
        "--num-threads", type=positive_int, default=8,
        help=(
            "Max parallelism. " +
            " Note that increasing this incurs a startup cost if using worktrees."))
    # TODO: make it a bool with a value instead of store_tree
    parser.add_argument(
        "--no-worktrees", action="store_true",
        help=(
            "By default, each thread runs the test in its own worktree. " +
            "Set this to disable that, and just run parallel tests in the main git tree"))
    # TODO: This cleanup logic is actually kinda stupid, probably would have
    # been better to just leave it to the uesr to prefix their test command with
    # a cleanup command if they care about it.
    parser.add_argument(
        "--no-cleanup-worktrees", action="store_true",
        help=(
            "By default, worktrees are hard-cleaned with git clean -fdx after each test." +
            "Set this to disable that. Ignored if --no-worktrees"))
    parser.add_argument("cmd", nargs="+")
    parser.add_argument(
        "--start", metavar="start", type=str, help=(
            "Start of bisection. If not provided, this tool assumes you've separately " +
            "begun a bisection and set this via 'git bisect good'"))
    parser.add_argument(
        "--end", type=str, help=(
            "Start of bisection. If not provided, either set this separately via " +
            "'git bisect bad', or this tool will use HEAD"))
    parser.add_argument(
        "--test-every-commit", action="store_true",
        help="Instead of bissecting, just test every commit and report every result.")

    return parser.parse_args()

if __name__ == "__main__":
    # Fix Python's threading system so that when a thread has an unhandled
    # exception the program exits.
    threading.excepthook = excepthook

    args = parse_args()
    was_in_bisect = True  # TODO lol
    if args.test_every_commit:
        if not args.start:
            print("--start is required with --test-every-commit")
            sys.exit(1)

        # TODO: run test_every_commit
        # results = test_every_commit(
        #     args.cmd, # TODO args lol
        #     num_threads=args.num_threads,
        #     use_worktrees=not args.no_worktrees,
        #     cleanup_worktrees=(not args.no_worktrees and
        #             not args.no_cleanup_worktrees))
        # # TODO: Print these as a range summary, or more like in a commit graph at least
        # for commit, exit_code in results:
        #     print(commit + ": exit code was " + str(exit_code))
    else:
        try:
            if not was_in_bisect:
                run_cmd(["git", "bisect", "start"])
            if args.start:
                run_cmd(["git", "bisect", "good", args.start])
            if args.end:
                run_cmd(["git", "bisect", "bad", args.to])
            # if "refs/bisect/bad" not in all_refs():
            #     run_cmd(["git", "bisect", "bad"])
            rev_range = "refs/bisect/good..refs/bisect/bad" # TODO
            result = dissect(rev_range, args.cmd,
                             num_threads=args.num_threads,
                             use_worktrees=not args.no_worktrees,
                             cleanup_worktrees=(not args.no_worktrees and
                                     not args.no_cleanup_worktrees))
            print("First bad commit is " + result)
        finally:
            if not was_in_bisect:
                run_cmd(["git", "bisect", "reset"])



# TODO
#
# See if it's possible to bisect in a worktree, so that the main tree can
#  be used meanwhile by the user.
#
# replacing args? Original idea was to have placeholders like with find
#  --exec. But can't remember why I thought this was useful.
#  One usage of this would be to implement something you could pass to make's
#  --jobserver-fd.
#
# capture output
#
# option to test based on tree? Worth checking if reverts result in the same tree.
#
# ability to watch a range like local CI
#
# look into async and see if we can drop the ugly thread pool
#
# remove requirement to use special bisect refs