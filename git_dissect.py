#!/usr/bin/python3

"""
Usage:
git bisect start
git bisect bad $BAD
git bisect good $GOOD
git-dissect [--[no-]worktrees] $*
"""

import dataclasses
import logging
import os
import signal
import subprocess
import tempfile
import threading
import queue

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

def good_refs() -> list[str]:
    all_refs = (l.split(" ")[1] for l in run_cmd(["git", "show-ref"]).splitlines())
    return [r for r in all_refs if r.startswith("refs/bisect/good")]

def rev_list(include: list[str], exclude: list[str], extra_args=[]) -> list[str]:
    args =  ["git", "rev-list"] + include + ["^" + r for r in exclude]
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

    def __init__(self, test_cmd, workdirs, cleanup_worktrees):
        # Used to synchronize enqueuement with the worker threads.
        self._cond = threading.Condition()
        self._rev_q = []
        # Separate mechanism for reporting results back. Keeping this separate
        # means that the only ones .wait()ing on self._cond are the worker
        # threads, so that it's safe for enqueue to use .notify instead of
        # notify_all.
        self._result_q = queue.Queue()
        self._threads = []
        self._subprocesses = {}
        self._dequeued = set()
        self._done = False
        self._test_cmd = test_cmd
        self._cleanup_worktrees = cleanup_worktrees

        # TODO: there's a bug leading to the same workdir getting reused by multiple threads.
        for workdir in workdirs:
            t = threading.Thread(target=self._work, args=(workdir,))
            t.start()
            self._threads.append(t)

    def enqueue(self, rev):
        with self._cond:
            if rev in self._dequeued:
                logger.info(f"Ignoring enqueue for already-dequeued revision {rev}")
                return

            self._rev_q.append(rev)
            logger.info(f"Enqueued {rev}, new queue depth {len(self._rev_q)}")
            self._cond.notify()

    def cancel(self, rev):
        with self._cond:
            if rev in self._subprocesses:
                # TODO: Does this work on Windows?
                self._subprocesses[rev].send_signal(signal.SIGINT)
            self._dequeued.add(rev)

    def _work(self, workdir):
        while True:
            with self._cond:
                while not self._rev_q and not self._done:
                    self._cond.wait()
                if self._done:
                    return

                rev = self._rev_q[0]
                self._rev_q = self._rev_q[1:]
                logger.info(f"Worker in {workdir} dequeued {rev}, " +
                            f"new queue depth {len(self._rev_q)}")

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
                    self._result_q.put((rev, 1))
                    continue

                self._subprocesses[rev] = p

            p.communicate()
            logger.info(f"Worker in {workdir} got result {p.returncode} for {rev}")
            self._result_q.put((rev, p.returncode))

    def wait(self):
        return self._result_q.get()

    def interrupt_and_join(self):
        with self._cond:
            for p in self._subprocesses.values():
                p.send_signal(signal.SIGINT)
            self._done = True
            self._cond.notify_all()
        for t in self._threads:
            t.join()

def do_dissect(args, pool):
    while True:
        # --revlist-all: Get hashes for bisection in order of precedence.
        # Returns bad as last entry
        revs = rev_list(include=["refs/bisect/bad"], exclude=good_refs(), extra_args="--bisect-all")
        if not revs:
            raise RuntimeError("Bug: found no revisions in bisect range")
        if len(revs) == 1:
            logger.info(f"Found single revision {revs[0]}, done")
            return revs[0]

        logger.info(f"Found {len(revs) - 1} remaining revisions to test")

        for rev in revs[1:]: # Drop "bad" commit
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

        # TODO: Here we're iterating on the range of all possible cancelable
        # commits; there might be a lot so it's probably faster (in the
        # worst case) to instead iterate on the enqueued commits and check
        # if each one is in the cancelable range.
        for commit in rev_list(include=cancel_to, exclude=cancel_from):
            pool.cancel(commit)

    return revs[0]

def excepthook(*args, **kwargs):
    threading.__excepthook__(*args, **kwargs)
    # Not sure exactly why sys.exit doesn't work here. This is cargo-culted from:
    # https://github.com/rfjakob/unhandled_exit/blob/e0d863a33469/unhandled_exit/__init__.py#L13
    os._exit(1)

def dissect(args, num_threads=8, use_worktrees=True, cleanup_worktrees=False):
    # Fix Python's threading system so that when a thread has an unhandled
    # exception the program exits.
    threading.excepthook = excepthook

    try:
        run_cmd(["git", "bisect", "log"])
    except CalledProcessError:
        raise NotBisectingError("Couldn't run 'git bisect log' - did you run 'git bisect'?")

    tmpdir = tempfile.mkdtemp()

    if use_worktrees:
        worktrees = [os.path.join(tmpdir, f"worktree-{i}") for i in range(num_threads)]
    else:
        worktrees = []

    pool = None
    try:
        # TODO: add option to skip worktrees
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

if __name__ == "__main__":
    print(run_cmd(["cat", "/etc/shadow"]))

# TODO
#   capture output
#   option to test based on tree? Worth checking if reverts result in the same tree.