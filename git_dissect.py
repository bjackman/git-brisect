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
            f'Stderr:\n{result.stderr.decode()}')
    return result.stdout.decode()

def all_refs() -> list[str]:
    return (l.split(" ")[1] for l in run_cmd(["git", "show-ref"]).splitlines())

def good_refs() -> list[str]:
    return [r for r in all_refs() if r.startswith("refs/bisect/good")]

def rev_list(include: list[str], exclude: list[str], extra_args=[]) -> list[str]:
    args =  ["git", "rev-list"] + include + ["^" + r for r in exclude] + extra_args
    out = run_cmd(args)
    return [l.split(" ")[0] for l in out.splitlines()]


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

    def cancel(self, rev):
        with self._cond:
            if rev in self._subprocesses:
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
                    self._out_q.append((rev, 1))
                    continue

                self._subprocesses[rev] = p

            p.communicate()
            logger.info(f"Worker in {workdir} got result {p.returncode} for {rev}")
            with self._cond:
                self._out_q.append((rev, p.returncode))
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

@dataclasses.dataclass
class RevRange:
    exclude: list[str]
    # Git bisect always has one "include" i.e. the single good ref. I don't
    # really know why there is 1 good ref but N bad ones. But we just follow
    # git's lead.
    include: str

def bisect_gen():
    """Generator for commits in the order we should test them"""
    # This is a breadth-first sinary search. Nodes are revision ranges. There's
    # an edge from each node to two sub-ranges, one for the commits before its
    # midpoint and one to the commits afterwards. The midpoint is as defined by
    # git's own bisect logic, basically it's something like "a commit whose
    # minimum distance from any of the edges of the range is largest".
    #
    # If you just think about a linear history, that all just sounds like a
    # really weird and complicaed way to describe a binary search. But thinking
    # of it this way means it also works for non-linear histories.
    #
    # A couple of silly things this algorithm does that illustrates why it's
    # suboptimal:
    # - If you have a linear bisection range of 10 commits and you take the
    #   first 2 of the outputs from this generator, you will get the 5th and 8th
    #   or something, where you actually want the 3rd and 7th, or something (I
    #   dunno I can't count). I.e. it's only optimal on linear ranges when you
    #   take 2^n-1 items from the generator.
    # - Say you have a bisection range that diverges into two branches that then
    #   re-merges, and you need to pick a single commit to test. Depending on
    #   the length of the branches, it's pretty likely that the optimal commit
    #   to test is the point of divergence or the merge commit. This algorithm
    #   is unlikely to pick that, instead it will probably pick a commit in the
    #   middle of whichever branch is longer.
    #
    # So I think a good next evolution for this algorithm would be something
    # like: pick N commits dividing up the current ranges such that the average
    # length of the child ranges is minimised. Set N to the number of idle
    # threads in our pool. Maybe computered scientists know how to do this.
    # Maybe it could be a nice Google interview question. But I would fail it.
    # Note that for all but the first invocation, N would be 1, so we'd
    # basically just be finding the longest existing range between two ongoing
    # tests, and kicking off a test for the midpoint of that range. But I don't
    # think that simplifies anything.
    #
    # Anyway, for now this algorithm is easy to implement and still quite fun
    # and its behaviour is stil not entirely ridiculous.
    ranges = [RevRange(exclude=good_refs(), include="refs/bisect/bad")]
    while ranges:
        r = ranges.pop(0)
        by_distance = rev_list(exclude=r.exclude, include=[r.include],
                               extra_args=["--bisect-all"])
        if len(by_distance) == 0:
            continue # Range is empty, we reached a leaf in our search
        midpoint = by_distance[0]
        yield midpoint
        # Add edge for range before the midpoint
        ranges.append(RevRange(exclude=r.exclude, include=midpoint + "^"))
        # And for range after.
        ranges.append(RevRange(exclude=r.exclude + [midpoint], include=r.include))

def do_dissect(args, pool):
    while True:
        gen = bisect_gen()
        # Start as many worker threads as possible, unless there's a reuslt
        # pending; that will influence which commits we need to test so there's
        # po point in adding new ones until we've processed it.
        while pool.in_q_length() < pool.num_threads and not pool.out_q_length():
            try:
                rev = next(gen)
            except StopIteration:
                # Nothing more to test, we must be done.
                return run_cmd(["git", "rev-parse", "refs/bisect/bad"])
            pool.enqueue(rev)

        result_commit, returncode = pool.wait()
        if returncode == 0:
            run_cmd(["git", "bisect", "good", result_commit])
            cancel_from = good_refs()
            cancel_to = [result_commit]
        else:
            run_cmd(["git", "bisect", "bad", result_commit])
            cancel_from = [result_commit]
            cancel_to = ["refs/bisect/bad"]
        logger.info("%s result was %d, cancelling %s to %s", result_commit, returncode,
                    cancel_from, cancel_to)
        for commit in rev_list(include=cancel_to, exclude=cancel_from):
            pool.cancel(commit)

    return revs[0]

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
    commits = rev_list(include=include, exclude=exclude)

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