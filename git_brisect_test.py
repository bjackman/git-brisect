#!/usr/bin/python3
from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime
import io
import inspect
import logging
import unittest
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading

from typing import Iterable, Optional

import hypothesis

import git_brisect

# Why does the test have a different style git wrapper function than the
# main code? It just does, OK! Stop asking annoying questions!
def git(*args: str):
    if os.path.exists("git_brisect_test.py"):
        raise RuntimeError("Cowardly refusing to stomp on my own git repo")

    res = subprocess.run(("git",) + args, capture_output=True)
    try:
        res.check_returncode()
    except:
        print(f"Git command {args} failed.")
        print(res.stderr.decode())
        raise
    return res.stdout.decode().strip()

def commit(msg: str, tag: Optional[str] = None):
    git("commit", "--allow-empty", "-m", msg)
    if tag is not None:
        git("tag", tag)
    return git("rev-parse", "HEAD")

def merge(msg: str, parents: list[str]):
    git("merge", "-m", msg, "--no-ff", *parents)
    return git("rev-pase", "HEAD")

class GitbrisectTest(unittest.TestCase):
    logger: logging.Logger

    def setUp(self):
        self.logger = logging.getLogger(self.id())

    def write_executable(self, path, content):
        """Write an executable and git add it"""
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o777) # SIR yes SIR o7

        git("add", path)

    def write_pass_fail_script(self, fail):
        """Write a script to path that can be used to "test" a commit.

        If fail, the script will exit with an error code.

        The script writes to a file in the current CWD (not the CWD the script
        runs in) that script_runs can use to detect when it gets run.

        This also git adds the script.
        """
        content = f"git rev-parse HEAD >> {os.getcwd()}/output.txt"
        if fail:
            content += "; exit 1"
        self.write_executable("run.sh", content)

    def read_stripped_lines(self, path):
        with open(path) as f:
            return [l.strip() for l in f.readlines()]

    def script_runs(self):
        """Return each of the commits tested by the script from write_script."""
        if not os.path.exists("output.txt"):
            return []
        return self.read_stripped_lines("output.txt")

class GitbrisectTestWithRepo(GitbrisectTest):
    def setUp(self):
        super().setUp()

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir))
        os.chdir(tmpdir)
        git("init")

class RevRangeTest(GitbrisectTestWithRepo):
    def setUp(self):
        super().setUp()

        # RevRange uses the repro to generate its __str__. Seems kinda sketchy
        # lol but it's useful. To make that work here, give it a repo.
        commit("foo", tag="foo")
        commit("bar", tag="bar")

    def test_from_string(self):
        cases = [
            ("foo", git_brisect.RevRange(exclude=[], include="foo")),
            ("foo ^bar", git_brisect.RevRange(exclude=["bar"], include="foo")),
            ("^bar foo", git_brisect.RevRange(exclude=["bar"], include="foo")),
            ("foo ^bar ^baz", git_brisect.RevRange(exclude=["bar", "baz"], include="foo")),
            ("bar..foo", git_brisect.RevRange(exclude=["bar"], include="foo")),
        ]
        for in_str, want in cases:
            with self.subTest(in_str=in_str):
                got = git_brisect.RevRange.from_string(in_str)
                self.assertEqual(want.include, got.include)
                self.assertEqual(want.exclude, got.exclude)

    def test_case_from_string_fail(self):
        cases = [
            "foo bar",
            # We could actually handle this in the case that foo is an ancestor
            # of bar or vice versa. But we don't.
            "bar...foo",
            "",
            "bar..",
            "..foo",
            "..foo^",
        ]
        for in_str in cases:
            with self.subTest(in_str=in_str):
                with self.assertRaises(git_brisect.BadRangeError):
                    got = git_brisect.RevRange.from_string(in_str)
                    self.logger.info(f"Got: {got}")

class TestBisection(GitbrisectTestWithRepo, unittest.IsolatedAsyncioTestCase):
    async def _run_worktree_test(self, use_worktrees: bool):
        """Run a test that should result in multiple worktrees being used.

        (Unless use_worktrees is false). The test script records what CWD it was
        run in, and leaves a file behind. It also records whether it found that
        file left over in the CWD from a previous run. Returns a list of pairs
        of (CWD, dirty), where "dirty" actually just means that the specific
        file was left behind.

        Does its best to ensure that at least one worktree gets reused, by
        having much more commits than threads. In theory this might actually
        flake though, if by chance the bisection algorithm lands on testing the
        exact 2 commits on either side of the breakage before any others.
        """
        script = f"""
            [ -e ./dirty ]; dirty_status=$?
            echo $(pwd) "$dirty_status" >> {os.getcwd()}/cwds.txt
            touch ./dirty
        """
        self.write_executable("log_cwd.sh", script)

        self.write_pass_fail_script(fail=False)
        init = commit("init")
        for i in range(50):
            commit("good " + str(i))
        self.write_pass_fail_script(fail=True)
        end = commit("end")

        self.logger.info(git("log", "--graph", "--all", "--oneline"))

        await git_brisect.brisect(f"{init}..{end}", ["sh", "./log_cwd.sh", "./run.sh"],
                                  use_worktrees=use_worktrees,
                                  out_dir=pathlib.Path.cwd(),
                                  num_threads=4)
        runs = self.script_runs()

        lines = self.read_stripped_lines("cwds.txt")
        # It's possible in theory that all but one of the threads was very slow,
        # and a single thread ended up doing all the work. And maybe by chance
        # it happened to test exactly the commits it needed, so we might just
        # get the same cwd twice. Seems unlikely so we don't bother to try and
        # avoid any case like this even though it might reduce test coverage.
        self.assertGreaterEqual(len(lines), 2)
        ret = []
        for line in lines:
            cwd, dirty_status = line.split(" ")
            self.assertIn(dirty_status, ("0", "1"))
            ret.append((cwd, dirty_status == "0"))
        return ret

    async def test_worktree_mode_no_cleanup(self):
        results = await self._run_worktree_test(use_worktrees=True)
        # If this flakes, it might be that by chance no worktree got reused,
        # which isn't technically a bug. Try to find a way to force them to get
        # reused.
        self.assertTrue(any(dirty for (cwd, dirty) in results))

    async def test_non_worktree_mode(self):
        results = await self._run_worktree_test(use_worktrees=False)

        # If nobody saw a dirty tree, seems like there was more parallelism than
        # expected, or we were racily cleaning up the tree.
        self.assertTrue(any(dirty for (cwd, dirty) in results))
        self.assertTrue(all(cwd == os.getcwd() for (cwd, dirty) in results))

    async def _test_thread_limit(self, num_threads):
        # At each commit will be a script that starts, touches a unique file
        # path, then hangs until a different unique filepath appers, then exits.
        commits = []
        for i in range(20):
            script = f"""
                touch {os.getcwd()}/started-{i}
                trap "touch {os.getcwd()}/done-{i}" EXIT
                while ! [ -e {os.getcwd()}/stop-{i} ]; do
                    sleep 0.01
                done
                exit {"1" if i > 17 else "0"}
            """
            self.write_executable("run.sh", script)
            commits.append(commit(msg=str(i), tag=str(i)))

        self.logger.info("\n" + git("log", "--graph", "--all", "--oneline"))

        # Run the brisection in the background
        brisect_task = asyncio.create_task(git_brisect.brisect(
            f"{commits[0]}..{commits[-1]}", args=["bash", "./run.sh"], num_threads=num_threads,
            out_dir=pathlib.Path.cwd()))

        # Gets the indexes of the commits whose test scripts are currently
        # started/done
        def tests_in_state(state):
            ret = set()
            for i in range(len(commits)):
                if os.path.exists(state + "-" + str(i)):
                    ret.add(i)
            return ret
        def running_tests():
            return tests_in_state("started") - tests_in_state("done")

        # Free up one thread at a time. We will do this at most once for each
        # remaining commit (i.e. excluding the ones whose tests are already
        # running)
        for i in range(len(commits) - num_threads):
            # No choice really but to just sleep for some random time, we can
            # detect when the expected tests get started, but we don't know when
            # we can trust that it doesn't also start any unexpected tests.
            await asyncio.sleep(0.5)

            self.assertLessEqual(len(running_tests()), num_threads,
                                 "after stopping %d threads" % i)

            if brisect_task.done():
                break

            # Let one of the threads exit
            to_terminate = next(iter(running_tests()))
            open("stop-" + str(to_terminate), "w").close()

        self.assertIsNotNone(await brisect_task)

    async def test_no_parallelism(self):
        await self._test_thread_limit(1)

    async def test_limited_threads(self):
        await self._test_thread_limit(4)

    # TODO:
    #
    #  pathological args? Like good and bad aren't ancestors?
    #
    #  ensuring script isn't run more times than necessary? Maybe done
    #
    #  ensuring test cleanups happen
    #
    #  gathering output
    #
    #  cancelation of tests that turn out to be unnecessary
    #
    #  ensure things don't break under long enqueuements

class TestRevRange(GitbrisectTest):
    def test_full_spec(self):
        # Hack: when specifying the range in full we don't actually need access
        # to the repo so we dont bother using "real" refs here.
        # This would fail if you tried to access "commits()" etc.

        r = git_brisect.RevRange.from_string("foo ^bar ^baz")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_brisect.RevRange.from_string("foo   ^bar") # multiple spaces
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar"])

        r = git_brisect.RevRange.from_string("^bar foo ^baz")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_brisect.RevRange.from_string("^bar ^baz foo")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_brisect.RevRange.from_string("foo")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, [])

        with self.assertRaises(git_brisect.BadRangeError):
            _ = git_brisect.RevRange.from_string("^foo")

        with self.assertRaises(git_brisect.BadRangeError):
            _ = git_brisect.RevRange.from_string("foo bar")

    def test_dot_dot(self):
        r = git_brisect.RevRange.from_string("foo..bar")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo"])

        r = git_brisect.RevRange.from_string("foo..bar ^baz")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo", "baz"])

        r = git_brisect.RevRange.from_string("^baz foo..bar")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo", "baz"])

        with self.assertRaises(git_brisect.BadRangeError):
            _ = git_brisect.RevRange.from_string("foo..bar..baz")

        with self.assertRaises(git_brisect.BadRangeError):
            _ = git_brisect.RevRange.from_string("foo..bar baz")


@dataclasses.dataclass
class DagNode:
    """DAG represented as a recursive structure"""
    # Non-negative identifier for the node. Nodes with smaller IDs are never
    # reachable from nodes with larger IDs.
    i: int
    parents: list[DagNode]

    # Includes the node itself.
    def ancestor_ids(self) -> list[int]:
      ancestors = set()
      def f(node):
          ancestors.add(node.i)
          for p in node.parents:
              f(p)
      f(self)
      # Sorting not functionally necessary. I have a very weak and fuzzy
      # suspicion it might improve Hypthoesis' search performance.
      return sorted(ancestors)

@dataclasses.dataclass(frozen=True)
class Dag:
    """DAG represented as a set of edges"""
    num_nodes: int
    edges: frozenset[tuple[int, int]] = dataclasses.field(default_factory=frozenset)

    def nodes(self) -> list[DagNode]:
        nodes = [DagNode(i=i, parents=[]) for i in range(self.num_nodes)]
        for (parent, child) in self.edges:
            nodes[child].parents.append(nodes[parent])
        return nodes

def uints_upto(n):
    return hypothesis.strategies.integers(min_value=0, max_value=n)

# Produces (i, j) where 0 <= i < j <= max_value
def distinct_sorted_uint_pairs(max_value: int) -> hypothesis.strategies.SearchStrategy:
    if max_value < 1:
        # hypothesis.strategies.integers would raise an exception due to
        # min_value>max_value.
        return hypothesis.strategies.nothing()
    return (
        uints_upto(max_value - 1)
        .flatmap(lambda n: hypothesis.strategies.tuples(
            hypothesis.strategies.just(n),
            hypothesis.strategies.integers(min_value=n+1, max_value=max_value)))
    )

# Strategy that takes input Dags and returns Dags that have the same set of
# nodes and one additional edge.
@hypothesis.strategies.composite
def with_additional_edge(draw, children):
    child = draw(children)
    new_edge = draw(
        # Edges must only go from "smaller" to "larger" nodes to maintain acyclicity.
        distinct_sorted_uint_pairs(max_value=child.num_nodes-1)
        .filter(lambda e: e not in child.edges)  # Drop duplicates
    )
    return Dag(num_nodes=child.num_nodes, edges=child.edges.union({new_edge}))

# Strategy that takes input Dags and returns Dags with one additional node,
# and an edge leading to that new node from one of the existing nodes.
@hypothesis.strategies.composite
def with_additional_node(draw, children):
    child = draw(children)
    new_node = child.num_nodes
    from_node = draw(uints_upto(child.num_nodes - 1))
    return Dag(num_nodes=child.num_nodes + 1, edges=child.edges.union({(from_node, new_node)}))

# Strategy that generates arbitrary DAGs.
dags = lambda: hypothesis.strategies.recursive(
    hypothesis.strategies.just(Dag(num_nodes=1)),
    lambda children: with_additional_edge(children) | with_additional_node(children),
    # Suggested by the hypothesis health checker. I don't really understand what
    # this does. I think part of the reaon this is confusing is that the
    # "leaves" of the strategy tree are actually the roots of that DAGs we're
    # generating? I dunno. How do I recursed graph?
    max_leaves=200)

# Strategy that generates a DAG and a pair of ("smaller", "larger") node IDs
# within that DAG.
@hypothesis.strategies.composite
def with_node_range(draw, dags):
    dag = draw(dags.filter(lambda d: d.num_nodes > 1))
    return (dag, draw(distinct_sorted_uint_pairs(max_value=dag.num_nodes-1)))

@dataclasses.dataclass
class BisectCase:
    dag: Dag
    # Range to bisect is this leaf and all its ancestors.
    leaf: int
    culprit: int

# Sets up a git repository in the CWD where hte history structure matches the
# given DAG. Each commit is tagged with the corresponding node's ID.
def create_history_cwd(dag: Dag):
    if os.path.exists(".git"):
        raise RuntimeError(f"Already a repository in {os.getcwd()}")

    git("init")

    # Maps DagNode.i to commit hash, faster than using git tags.
    commits: dict[int, str] = {}
    def create_commit(node: DagNode):
        if node.i in commits:
            return

        if len(node.parents) == 0:
            # This is the "root" of the history. In this code we always have
            # exactly one of these - check that.
            assert len(commits) == 0

        for parent in node.parents:
            create_commit(parent)

        if node.parents:
            # https://stackoverflow.com/questions/25384469/when-a-git-commit-has-multiple-parents-what-are-the-stats-calculated-against
            # If the parent commits are already ancestors of HEAD then git merge
            # doesn't create a commit, even with --no-ff (--no-ff means create a
            # merge commit even if HEAD is an ancestor of the input revision).
            # So we need to check out an ancestor-most parent. Ancestry is a
            # sub-ordering of the numeric ordering of the node IDs so min will
            # do the trick.
            git("checkout", str(min(p.i for p in node.parents)))
        if len(node.parents) <= 1:
            commits[node.i] = commit(msg=str(node.i))
        else:
            commits[node.i] = merge(msg=str(node.i),
                                    parents=[str(p.i) for p in node.parents])
        git("tag", str(node.i))

    # Actually we only need to do this for the leaves but we don't have those to
    # hand at the moment.
    for node in dag.nodes():
        create_commit(node)

# Factory strategy that generates a DAG and the ID of a node within that
# DAG, and a leaf node that is reachable from that other node.
@hypothesis.strategies.composite
def bisect_cases(draw, dags):
    dag = draw(dags)
    leaves = set(range(dag.num_nodes))
    for src, _ in dag.edges:
        if src in leaves:
            leaves.remove(src)
    leaf_id = sorted(leaves)[draw(uints_upto(len(leaves) - 1))]
    leaf_ancestors = dag.nodes()[leaf_id].ancestor_ids()
    ancestor_id = leaf_ancestors[draw(uints_upto(len(leaf_ancestors) - 1))]
    return BisectCase(dag=dag, culprit=ancestor_id, leaf=leaf_id)

class TestWithHypothesis(GitbrisectTest):
    repo_cache: dict[Dag, str] = {}  # Maps DAGs to repo paths

    def setUp(self):
        super().setUp()

    def setup_repo(self, dag: Dag):
        if path := self.repo_cache.get(dag):
            os.chdir(path)
            return

        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        create_history_cwd(dag)
        self.addClassCleanup(lambda: shutil.rmtree(tmpdir))

    def describe(self, rev: str) -> str:
        return git("describe", "--tags", rev).strip()

    # Asserts that subsets are disjoint and that their union is superset.
    # Assumes that the sets are of git revisions that can be `git describe`d,
    # i.e. they were set up by setup_repo so they have tags.
    def assertCommitSetPartition(self, subsets: Iterable[set[str]], superset: set[str]):
        # "describe" the commits; because we tagged each commit with the ID of
        # the corresponding DAG node this produces error messages that are
        # helpful when looking at the example inputs.
        subsets = list(map(lambda s: set(map(self.describe, s)), subsets))
        superset = set(map(self.describe, superset))

        self.assertSetEqual(set().union(*subsets), superset)
        subsets_size = sum(len(s) for s in subsets)
        if subsets_size < len(superset):  # Above should already have failed?
            raise RuntimeError("errrrrrrr ummmmmm bug in the test logic!")
        # If subsets_size > len(superset) then they must not be disjoint.
        self.assertEqual(subsets_size, len(superset))

    # Hypothesis doesn't know how to run async methods, this defines an Executor
    # (Hypothesis concept) to work around that issue.
    def execute_example(self, test_func):
        result = test_func()
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        if callable(result):
            result = result()
        return result

    @hypothesis.given(dag_and_range=with_node_range(dags()))
    @hypothesis.example((Dag(num_nodes=4, edges=frozenset({(0, 1), (0, 2), (2, 3), (1, 3)})),
                         (2, 3)))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=1))
    def test_range_split(self, dag_and_range: tuple[Dag, tuple[int, int]]):
        (dag, (exclude_node, include_node)) = dag_and_range
        self.setup_repo(dag)
        rev_range = git_brisect.RevRange(exclude=[str(exclude_node)],
                                         include=str(include_node))
        m = rev_range.midpoint()
        if not m:
            return # Range is empty
        before, after = rev_range.split(m)
        self.assertCommitSetPartition((before.commits(), after.commits()), rev_range.commits())

        # Same for dropping the tip of the range.
        subranges = rev_range.drop_include()
        self.assertCommitSetPartition((s.commits() for s in subranges),
                                      rev_range.commits() - {git_brisect.rev_parse(rev_range.include)})

    @hypothesis.given(case=bisect_cases(dags()))
    # Some random examples that detected bugs in the past
    @hypothesis.example(BisectCase(
        dag=Dag(num_nodes=3, edges=frozenset({(0, 1), (0, 2), (1, 2)})),
        culprit=2, leaf=2))
    @hypothesis.example(BisectCase(
        dag=Dag(num_nodes=4, edges=frozenset([(0, 1), (1, 2), (2, 3)])),
        culprit=2, leaf=3))
    @hypothesis.example(BisectCase(
        dag=Dag(num_nodes=4, edges=frozenset({(0, 1), (1, 2), (0, 2), (0, 3)})),
        leaf=2, culprit=2))
    @hypothesis.example(BisectCase(
        dag=Dag(num_nodes=4, edges=frozenset({(0, 1), (1, 2), (2, 3), (0, 3)})),
        leaf=3, culprit=3))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=1))
    async def test_bisect(self, case: BisectCase):
        self.logger.info(f"Running {case}")
        self.setup_repo(case.dag)
        # Bisect the range consisting of all the ancestors of the leaf node.
        range_spec = str(case.leaf)

        # tested_commits.txt will contain a line with each commit that was
        # tested. Create it in advance to simplify the case that no tests get
        # run.
        tested_commits_path = f"{os.getcwd()}/tested_commits.txt"
        with open(tested_commits_path, "w") as f:
            pass
        self.addCleanup(lambda: os.remove(tested_commits_path))

        # Simulate a bug being introduced by our culprit commit. Commits are
        # tagged by the ID of the node they are generated from. We use that tag
        # to test if the culprit is an ancestor of HEAD (this command considers
        # commits to be their own ancestor); if it is then HEAD is "broken".
        cmd = (f"git describe --tags HEAD >> {os.getcwd()}/tested_commits.txt; " +
               f"! git merge-base --is-ancestor {case.culprit} HEAD")
        result = await git_brisect.brisect(rev_range=range_spec, args=["bash", "-c", cmd],
                                           out_dir=pathlib.Path.cwd())
        self.assertEqual(self.describe(result), str(case.culprit))

        with open(tested_commits_path) as f:
            tested_commits = f.readlines()
        tested_multiple = [v for v, c in collections.Counter(tested_commits).items() if c > 1]
        self.assertFalse(tested_multiple)  # Should be empty (nothing tested twice)

class TestEndToEnd(unittest.IsolatedAsyncioTestCase):
    logger: logging.Logger

    now = datetime.datetime(2023, 10, 21, 17, 30, 4, 605908)

    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir))
        os.chdir(tmpdir)
        self.logger = logging.getLogger(self.id())
        create_history_cwd(Dag(edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 4)}), num_nodes=5))

    def grep(self, s: str, pattern: str):
        ret = None
        regexp = re.compile(pattern)
        for line in s.splitlines():
            if regexp.match(line):
                self.assertIsNone(ret, f"Multiple lines match /{pattern}/:\n{line}\n{ret}")
                ret = line
        if not ret:
            raise AssertionError(f"Didn't find a line matching /{pattern}/")
        return ret

    def test_bisect_smoke(self):
        # "Bug" is in commit 2.
        c = "! git merge-base --is-ancestor 2 {}"
        cmd = ["bash", "-c", c.format("HEAD")]
        cmd_no_head = ["bash", "-c", c.format("$GIT_BRISECT_TEST_REVISION")]

        args_cases = [
            ["git-brisect", "0..4", "--"] + cmd,
            ["git-brisect", "4 ^0", "--"] + cmd,
            ["git-brisect", "--no-worktrees", "0..4", "--"] + cmd_no_head,
            ["git-brisect", "--num-threads", "4", "0..4", "--"] + cmd,
            ["git-brisect", "-n", "4", "0..4", "--"] + cmd,
            ["git-brisect", "-n4", "0..4", "--"] + cmd,
        ]
        for args in args_cases:
            with self.subTest(args=args):
                self.logger.info(args)
                stdout = io.StringIO()
                asyncio.run(git_brisect.main(args, stdout, self.now))
                commit_hash = git_brisect.rev_parse("2")
                line = self.grep(stdout.getvalue(), '^First bad commit is.*')
                self.assertEqual(line, f'First bad commit is {commit_hash} ("2")')

    async def test_out_dir_creation(self):
        async def check_out_dir(args):
            stdout = io.StringIO()
            full_args = ["git-brisect"] + args + [
                "0..4", "--", "bash", "-c", "! git merge-base --is-ancestor 2 HEAD"]
            await git_brisect.main(full_args, stdout, self.now)
            line = self.grep(stdout.getvalue(), "^Writing output to.*")
            out_dir = line.split(" ")[-1]
            self.assertTrue(os.path.isdir(out_dir))
            self.addCleanup(lambda: shutil.rmtree(out_dir))
            return out_dir

        # Default: just check existence.
        await check_out_dir([])

        out_dir = await check_out_dir(["--out-dir", f"{os.getcwd()}/foo-bar"])
        self.assertEqual(out_dir, f"{os.getcwd()}/foo-bar")

        out_dir_1 = await check_out_dir(["--out-dir-in", f"{os.getcwd()}/foo-bar"])
        self.assertTrue(out_dir_1.startswith(f"{os.getcwd()}/foo-bar/"))

        # Second time, since the timestamp is the same, the output directory
        # should get a unique suffix.
        out_dir_2 = await check_out_dir(["--out-dir-in", f"{os.getcwd()}/foo-bar"])
        self.assertTrue(out_dir_2.startswith(out_dir_1))
        self.assertNotEqual(out_dir_2, out_dir_1)

    async def test_out_dir_contents(self):
        out_dir = pathlib.Path.cwd() / "out_dir"
        out_dir.mkdir()

        cmd = ["bash", "-c", f"""
            tag=$(git describe --tags HEAD)
            echo "hello from "$tag" stdout"
            echo "hello from "$tag" stderr" 1>&2
            echo "hello from "$tag" output" > $GIT_BRISECT_OUTPUT_DIR/t.txt
            case "$tag" in
                0|1)
                    exit 0
                    ;;
                2)
                    # Commit 2 is the culprit. To guarantee that 3's test gets
                    # started before we can determine the culprit, its test
                    # hanges until a magic file is touched.
                    while [ ! -f {os.getcwd()}/unhang ]; do
                        sleep 0.1
                    done
                    exit 1
                    ;;
                3)
                    read    # Commit 3's test runs forever
                    ;;
            esac
        """]

        # Run the brisection in the background
        args = ["git-brisect", "--out-dir", str(out_dir), "--num-thread", "4", "0..4", "--"] + cmd
        brisect_task = asyncio.create_task(git_brisect.main(args, io.StringIO(), self.now))


        # Wait for test 3's output to exist so we know that test got started
        # before the culprit was detected, so we can assert its cancelation is
        # reported.
        test_3_output = out_dir / git_brisect.rev_parse("3") / "output" / "t.txt"
        while not brisect_task.done() and not test_3_output.exists():
            await asyncio.sleep(0.1)
        self.assertTrue(test_3_output.exists())

        # Unblock test 2; now the whole bisection should complete.
        with open("unhang", "w") as f:
            pass
        await brisect_task

        # Note we can't really assert on comit 0's output. But 1 and 2 must have
        # been tested in order to detect the culprit, and we have carefully
        # arranged things so that 3 must also get tested.
        for rev, sub_path, want in [
            ("1", "stdout.txt", "hello from 1 stdout\n"),
            ("1", "stderr.txt", "hello from 1 stderr\n"),
            ("1", "output/t.txt", "hello from 1 output\n"),
            ("1", "returncode.txt", "0"),
            ("2", "stdout.txt", "hello from 2 stdout\n"),
            ("2", "stderr.txt", "hello from 2 stderr\n"),
            ("2", "output/t.txt", "hello from 2 output\n"),
            ("2", "returncode.txt", "1"),
            ("3", "CANCELED", ""),
        ]:
            path = out_dir / git_brisect.rev_parse(rev) / sub_path
            with path.open() as f:
                got = f.read()
                self.assertEqual(got, want, f"For path: {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)-6s %(message)s")
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    unittest.main()