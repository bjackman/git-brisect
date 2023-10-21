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
import pathlib
import signal
import subprocess
import tempfile
import threading
import typing
import queue
import shutil
import sys

from typing import Optional, TextIO
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

# First line of commit message (git calls this the "subject" but that's
# confusing).
def commit_title(rev):
    output = run_cmd(["git", "log", "-n1", "--format=%s", rev])
    if output.endswith("\n"):
        output = output[:-1]
    return output


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
    def from_string(cls, s: str):
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
        # Trying to debug rare test flake where we somehow dissect the wrong
        # range. The dissection works fine but the range printed at the
        # beginning of do_dissect is wrong.
        logger.debug(f"RevRange.from_string({s}): include={include} exclude={exclude}")
        return cls(exclude=exclude, include=include[0])

    def __repr__(self):
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
            raise RuntimeError(f"{describe(commit)} not in {self}")

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

    def __init__(self, test_cmd: Iterable[str], workdirs: Iterable[pathlib.Path]):
        # Used to synchronize enqueuement with the worker threads.
        self._cond = threading.Condition()
        self._in_q = []
        self._out_q = []
        self._threads = []
        self._subprocesses = {}
        self._done = False
        self._test_cmd = test_cmd

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

    def _work(self, workdir: pathlib.Path):
        while True:
            with self._cond:
                while not self._in_q and not self._done:
                    self._cond.wait()
                if self._done:
                    return

                rev = self._in_q[0]
                self._in_q = self._in_q[1:]

                # TODO: Capture stdout and stderr somewhere useful.
                run_cmd(["git", "-C", str(workdir), "checkout", rev])
                try:
                    p = subprocess.Popen(
                        self._test_cmd, cwd=workdir,
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
    logger.info(f"Dissecting range {full_range}")
    # Note: a "culprit" is a commit that is bad and all of whose ancestors are
    # good. There may actually be more than one such commit, but sometimes it's
    # clearer to talk about "the" culprit anyway, that just means the one we're
    # eventually gonna return.

    # full_range is the range that contains the culprit. Our goal is to shrink
    # it until it contains only one commit, which must be the culprit.
    # untested_ranges contains non-overlapping ranges of commits that we haven't
    # yet kicked off tests for. The tip commit has already been tested (as bad)
    # by the user, hence drop_include().
    untested_ranges = full_range.drop_include()
    while len(full_range.commits()) > 1:
        # Start as many worker threads as possible, unless there's a result
        # pending; that will influence which commits we need to test so there's
        # po point in adding new ones until we've processed it. Here we do a
        # breadth-first bisection of the overall range to divide it among
        # available threads. That's not a fully optimal distribution; we want
        # some algorithm that minimizes the average distance between the commits
        # we test. But I can't think of such an algorithm; this one is easy to
        # implement, still kinda fun, and produces not completely insane
        # behaviour.
        while untested_ranges and pool.num_pending() < pool.num_threads and not pool.out_q_length():
            r = untested_ranges.pop(0)
            if not r.commits():
                continue

            # Kick off a test
            pool.enqueue(r.midpoint())
            before, after = r.split(r.midpoint())
            untested_ranges += before.drop_include()
            untested_ranges.append(after)

            # The commit we'll learn most by testing is the midpoint of the
            # largest remaining subrange. Sorting by size makes this a sort of
            # hill-climbing search.
            untested_ranges = sorted(untested_ranges, key=lambda r: len(r.commits()), reverse=True)

        result_commit, returncode = pool.wait()
        if result_commit not in full_range.commits():
            continue  # We cancelled this one - result is not interesting or meaningful.
        logger.info(f"Got result {returncode} for {describe(result_commit)}")
        logger.info(f"    Considering range {full_range}")

        if returncode == 0:  # There is a culprit that is not an ancestor of this commit.
            (cancel, full_range) = full_range.split(result_commit)
        else:                # This commit or one of its ancestors is a culprit.
            (full_range, cancel) = full_range.split(result_commit)

        logger.info(f"    Canceling {cancel}, remaining: {full_range}")
        pool.cancel(cancel)

        # Drop known-* commits from future testing. subranges are either subsets
        # of cancel, disjoint from it. (If one wasn't, that must imply we kicked
        # off a test for a commit within it, in which case we should have split
        # it up).
        untested_ranges = [s for s in untested_ranges if s.midpoint() in full_range.commits()]

    return rev_parse(full_range.include)

def excepthook(*args, **kwargs):
    threading.__excepthook__(*args, **kwargs)  # pytype: disable=module-attr
    # Not sure exactly why sys.exit doesn't work here. This is cargo-culted from:
    # https://github.com/rfjakob/unhandled_exit/blob/e0d863a33469/unhandled_exit/__init__.py#L13
    os._exit(1)

def dissect(rev_range: str, args: Iterable[str], num_threads=8, use_worktrees=True):
    tmpdir = None
    if use_worktrees:
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        worktrees = [tmpdir / f"worktree-{i}" for i in range(num_threads)]
    else:
        worktrees = []

    pool = None
    try:
        logger.info("Setting up worktrees...")
        for w in worktrees:
            run_cmd(["git", "worktree", "add", w, "HEAD"])
        logger.info("...Done setting up worktrees.")
        pool = WorkerPool(args,
                          worktrees or [pathlib.Path.cwd() for _ in range(num_threads)])
        return do_dissect(args, pool, RevRange.from_string(rev_range))
    finally:
        if pool:
            logger.info("Aborting pending tests...")
            pool.interrupt_and_join()
            logger.info("...Pending tests aborted.")
        logger.info("Tearing down worktrees...")
        for w in worktrees:
            run_cmd(["git", "worktree", "remove", "--force", w])
        logger.info("... Done tearing down worktrees.")
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
                      num_threads=8, use_worktrees=True):
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
                          worktrees or [os.getcwd() for _ in range(num_threads)])
        return do_test_every_commit(pool, commits)
    finally:
        if pool:
            pool.interrupt_and_join()
        for w in worktrees:
            run_cmd(["git", "worktree", "remove", "--force", w])
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="git bisect and rebase --exec, but with parallelism")
    # TODO: add short args (not sure how to do this, no docs on the plane!)
    def positive_int(val):
        i = int(val)
        if i < 0:
            raise ValueError("value cannot be negative")
        return i
    parser.add_argument(
        "-n", "--num-threads", type=positive_int, default=8,
        help=(
            "Max parallelism. " +
            " Note that increasing this incurs a startup cost if using worktrees."))
    parser.add_argument(
        "--no-worktrees", action="store_true",
        help=(
            "By default, each thread runs the test in its own worktree. " +
            "Set this to disable that, and just run parallel tests in the main git tree"))
    parser.add_argument("range", help=(
        'Commit range to bisect, using syntax described in SPECIFYING RANGES ' +
        'section of gitrevisions(7), but without the ... syntax option. Must have ' +
        'a single "included" commit soo "foo ^bar ^baz" is valid but "foo bar ' +
        '^baz" is not. The "included" commit (foo in the example) is the one you ' +
        'know is "bad". The excluded commits (^bar and ^baz in the examples) are ' +
        'ones you know are "good".'
    ))
    parser.add_argument("cmd", nargs="+")

    return parser.parse_args(argv)

def main(argv: list[str], output: TextIO) -> int:
    args = parse_args(argv[1:])
    if False:  # args.test_every_commit:
        raise NotImplementedError("soz")
    else:
        result = dissect(args.range, args.cmd,
                         num_threads=args.num_threads,
                         use_worktrees=not args.no_worktrees)
        output.write(f'First bad commit is {result} ("{commit_title(result)}")')
    return 0

if __name__ == "__main__":
    # Fix Python's threading system so that when a thread has an unhandled
    # exception the program exits.
    threading.excepthook = excepthook

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)-6s %(message)s")

    sys.exit(main(sys.argv, sys.stdout))

# TODO
#
# Support reusing existing worktrees
#
# Handle Ctrl-C and ensure worktrees are cleaned up
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
