#!/usr/bin/python3

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
import queue
import shutil
import sys

from typing import Optional
from collections.abc import Iterable

logger = logging.getLogger(__name__)

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
            f'Stderr:\n{result.stderr.decode()}\nStdout:\n{result.stdout.decode()}')
    return result.stdout.decode()

def all_refs() -> list[str]:
    return (l.split(" ")[1] for l in run_cmd(["git", "show-ref"]).splitlines())

def good_refs() -> list[str]:
    return [r for r in all_refs() if r.startswith("refs/bisect/good")]

def rev_parse(rev):
    return run_cmd(["git", "rev_parse", rev]).strip()

class RevRange:
    # Git bisect always has one "include" i.e. the single good ref. I don't
    # really know why there is 1 good ref but N bad ones. But we just follow
    # git's lead. I thik this would probably be obvious if you tried to
    # implement this algorithm in a way where both ends of the range can be
    # plural?
    def __init__(self, exclude: Iterable[str], include: str):
        self.exclude = exclude
        self.include = include

        self._commits: Optional[set[str]] = None
        self._midpoint: Optional[str] = None

    def __str__(self):
        return "RevRange([%s] %d commits)" % (self._spec(), len(self.commits()))

    def _spec(self):
        """Args to be passed to git rev-list to describe the range"""
        return [self.include] + ["^" + r for r in self.exclude]

    def _get_commits(self):
        if self._commits is not None:
            return

        # Find commits in range, sorted by descending distance to nearest
        # endpoint.
        args =  ["git", "rev-list", "--bisect-all"] + self._spec()
        out = run_cmd(args)
        commits_by_distance = [l.split(" ")[0] for l in out.splitlines()]

        if len(commits_by_distance):
            self._midpoint = commits_by_distance[0]
        self._commits = set(commits_by_distance)

    def midpoint(self) -> str:
        self._get_commits()
        return self._midpoint

    def commits(self) -> set[str]:
        self._get_commits()
        return self._commits

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
        self._dequeued = set()
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
            if rev in self._dequeued:
                logger.info(f"Ignoring enqueue for already-dequeued revision {rev}")
                return

            self._in_q.append(rev)
            logger.info(f"Enqueued {rev}, new queue depth {len(self._in_q)}")
            # TODO: Because we use the same condition variable for input and
            # output, we need notify_all. Rework this to avoid that.
            self._cond.notify_all()

    def in_q_length(self):
        with self._cond:
            return len(self._in_q)

    def out_q_length(self):
        with self._cond:
            return len(self._out_q)

    def cancel(self, range: RevRange):
        with self._cond:
            for rev, p in self._subprocesses.items():
                if rev in range.commits():
                    # TODO: Does this work on Windows?
                    self._subprocesses[rev].send_signal(signal.SIGINT)
                    self._dequeued.add(rev)

    def _work(self, workdir):
        while True:
            with self._cond:
                while not self._in_q and not self._done:
                    self._cond.wait()
                if self._done:
                    return

                rev = self._in_q[0]
                self._in_q = self._in_q[1:]
                logger.info(f"Worker in {workdir} dequeued {rev}, " +
                            f"new queue depth {len(self._in_q)}")

                if rev in self._dequeued:
                    logger.info(f"Worker in {workdir} ignoring {rev}, already dequeued")
                    continue
                self._dequeued.add(rev)

            # TODO: Capture stdout and stderr somewhere useful.
            run_cmd(["git", "-C", workdir, "checkout", rev])
            if self._cleanup_worktrees:
                run_cmd(["git", "-C", workdir, "clean", "-fdx"])
            try:
                p = subprocess.Popen(
                    self._test_cmd, cwd=workdir,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (PermissionError, FileNotFoundError) as e:
                logging.info(f"Error running at {rev} ({e}), returning code 1")
                with self._cond:
                    self._out_q.append((rev, 1))
                    self._cond.notify_all()
                    continue

                self._subprocesses[rev] = p

            p.communicate()
            with self._cond:
                self._out_q.append((rev, p.returncode))
                logger.debug(f"Worker in {workdir} got result {p.returncode} for {rev} " +
                             f"(new out_q lenth {len(self._out_q)}")
                self._cond.notify_all()

    def wait(self):
        with self._cond:
            while len(self._out_q) == 0:
                self._cond.wait()
            ret = self._out_q[0]
            self._out_q = self._out_q[1:]
            logger.debug(f"Popped from out_q, new length {len(self._out_q)}")
            return ret

    def interrupt_and_join(self):
        with self._cond:
            for p in self._subprocesses.values():
                p.send_signal(signal.SIGINT)
            self._done = True
            self._cond.notify_all()
        for t in self._threads:
            t.join()

def do_dissect(args, pool):
    # Total range of commits that might contain the bug.
    full_range = RevRange(exclude=good_refs(), include=rev_parse("refs/bisect/bad"))
    # Subranges between commits that we've kicked off tests for. Used to
    # prioritise commits for testing.
    subranges = [full_range]
    while len(full_range.commits()) > 1:
        # Start as many worker threads as possible, unless there's a reuslt
        # pending; that will influence which commits we need to test so there's
        # po point in adding new ones until we've processed it. Here we do a
        # breadth-first bisection of the overall range to divide it among
        # available threads. That's not a fully optimal distribution; we want
        # some algorithm that minimizes the average distance between the commits
        # we test. But I can't think of such an algorithm; this one is easy to
        # implement, still kinda fun, and produces not completely insane
        # behaviour.
        while subranges and pool.in_q_length() < pool.num_threads and not pool.out_q_length():
            r = subranges.pop(0)
            midpoint = r.midpoint()
            pool.enqueue(midpoint)

            # The midpoint divided the range into two subranges, add them to the
            # queue unless the are empty.
            #
            # TODO: The two ranges we generate here can overlap. This can lead
            # to us trying to test the same commit twice, or generally just
            # picking commits sub-optimally. Should find a way to drop the
            # common commits from one of the ranges. I think maybe git
            # merge-base could work here?
            after = RevRange(exclude=r.exclude, include=midpoint + "^")
            if after.commits():
                subranges.append(after)
            before = RevRange(exclude=r.exclude + [midpoint], include=r.include)
            if before.commits():
                subranges.append(before)

        result_commit, returncode = pool.wait()
        if returncode == 0:
            # We'll cancel all the tests of commits that are about to be
            # subsumed by the refs/bisect/good* ref we add.
            cancel = RevRange(exclude=good_refs(), include=result_commit)
            # TODO: now we have a proper algorithm, running the git bisect
            # commands is pointless.
            run_cmd(["git", "bisect", "good", result_commit])
        else:
            # We'll cancel everything that isn't an ancestor of the new
            # refs/bisect/bad we're about to set.
            bad = rev_parse("refs/bisect/bad")
            cancel = RevRange(exclude=good_refs() + [result_commit], include=bad)
            run_cmd(["git", "bisect", "bad", result_commit])
        logger.info("%s result was %d, cancelling %s", result_commit, returncode, cancel)
        pool.cancel(cancel)

        # Update all the ranges we need to consider testing.
        new_ranges = []
        for r in subranges:
            if returncode == 0:
                new = RevRange(exclude=r.exclude + [result_commit], include=r.include)
            else:
                new = RevRange(exclude=r.exclude, include=result_commit)
            if new.commits():
                new_ranges.append(new)
        # Sort by size of the range - this is like a best-first search.
        subranges = sorted(new_ranges, key=lambda r: len(r.commits()), reverse=True)

        full_range = RevRange(exclude=good_refs(), include=rev_parse("refs/bisect/bad"))

    return full_range.midpoint()

def excepthook(*args, **kwargs):
    threading.__excepthook__(*args, **kwargs)
    # Not sure exactly why sys.exit doesn't work here. This is cargo-culted from:
    # https://github.com/rfjakob/unhandled_exit/blob/e0d863a33469/unhandled_exit/__init__.py#L13
    os._exit(1)

def in_bisect() -> bool:
    try:
        run_cmd(["git", "bisect", "log"])
    except CalledProcessError:
        return False
    else:
        return True

def dissect(args, num_threads=8, use_worktrees=True, cleanup_worktrees=False):
    if not in_bisect():
        raise NotBisectingError("Couldn't run 'git bisect log' - did you run 'git bisect'?")

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
        return do_dissect(args, pool)
    finally:
        if pool:
            pool.interrupt_and_join()
        for w in worktrees:
            run_cmd(["git", "worktree", "remove", "--force", w])
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

@dataclasses.dataclass
class Range:
    start: str
    end: str
    broken: bool

def do_test_every_commit(pool, commits):
    for commit in commits:
        pool.enqueue(commit)

    results = []
    for _ in range(len(commits)):
        results.append(pool.wait())

    return results

# include and exclude specify the set of commits to test.
def test_every_commit(args, include: list[str], exclude: list[str],
                      num_threads=8, use_worktrees=True, cleanup_worktrees=False):
    commits = RevRange(include=include, exclude=exclude).commits()

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
    was_in_bisect = in_bisect()
    if args.test_every_commit:
        if not args.start:
            print("--start is required with --test-every-commit")
            sys.exit(1)

        # TODO: Cleanup the way we specify the range to test, more like git rev-list.
        results = test_every_commit(
            args.cmd, include=[args.end or "HEAD"], exclude=[args.start + "^"],
            num_threads=args.num_threads,
            use_worktrees=not args.no_worktrees,
            cleanup_worktrees=(not args.no_worktrees and
                    not args.no_cleanup_worktrees))
        # TODO: Print these as a range summary, or more like in a commit graph at least
        for commit, exit_code in results:
            print(commit + ": exit code was " + str(exit_code))
    else:
        try:
            if not was_in_bisect:
                run_cmd(["git", "bisect", "start"])
            if args.start:
                run_cmd(["git", "bisect", "good", args.start])
            if args.end:
                run_cmd(["git", "bisect", "bad", args.to])
            if "refs/bisect/bad" not in all_refs():
                run_cmd(["git", "bisect", "bad"])
            result = dissect(args.cmd,
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