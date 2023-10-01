#!/usr/bin/python3
from __future__ import annotations

import dataclasses
import datetime
import logging
import unittest
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading

from typing import Iterable

import hypothesis

import git_dissect

class GitDissectTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(self.id())

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir))
        os.chdir(tmpdir)

        self.commit_counter = 0

        self.git("init")

    # Why does the test have a different style git wrapper function than the
    # main code? It just does, OK! Stop asking annoying questions!
    def git(self, *args):
        res = subprocess.run(("git",) + args, capture_output=True)
        try:
            res.check_returncode()
        except:
            print(f"Git command {args} failed.")
            print(res.stderr.decode())
            raise
        return res.stdout.decode().strip()

    def commit(self, msg=None):
        if msg is None:
            msg = f"commit {self.commit_counter}"
            self.commit_counter += 1
        self.git("commit", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def merge(self, *parents):
        self.git("merge", "--no-edit", *parents)
        return self.git("rev-pase", "HEAD")

    def write_executable(self, path, content):
        """Write an executable and git add it"""
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o777) # SIR yes SIR o7

        self.git("add", path)

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

class TestBisection(GitDissectTest):
    def test_no_tests_needed(self):
        # TODO: Actually we should check that the script doesn't get run.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        self.write_pass_fail_script(fail=True)
        bad = self.commit()

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(f"{good}..{bad}", ["sh","run.sh"]), bad)
        self.assertFalse(os.path.exists("output.txt"), "Script was run unnecessarily")

    def test_smoke(self):
        # Linear history where the code gets broken in the middle.
        self.write_pass_fail_script(fail=False)
        good = self.commit()
        want_test1 = self.commit()
        self.write_pass_fail_script(fail=True)
        want_culprit = self.commit()
        bad = self.commit()

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(f"{good}..{bad}", ["sh", "./run.sh"]), want_culprit)
        self.assertCountEqual(self.script_runs(),
                              [want_test1, want_culprit],
                              "didn't get the expected set of script runs")

    def test_nonlinear_multiple_good(self):
        # Branched history where the code is good in both branches and then broken after a merge
        self.write_pass_fail_script(fail=False)
        base = self.commit("base")
        good1 = self.commit("good1")
        self.git("checkout", base)
        good2 = self.commit("good2")
        merge = self.merge(good1)
        self.write_pass_fail_script(fail=True)
        want = self.commit("want")
        bad = self.commit("bad")

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(f"{good1}..{bad}", ["sh", "./run.sh"]), want)
        runs = self.script_runs()
        self.assertIn(want, runs)
        self.assertIn(merge, runs)

    def test_bug_in_branch(self):
        # Branched history where the bug arises in one of the branches.
        self.write_pass_fail_script(fail=False)
        good = self.commit("good")
        other = self.commit("other")
        self.git("checkout", good)
        self.write_pass_fail_script(fail=True)
        want = self.commit("want")
        also_test = self.commit("also_test")
        self.git("checkout", other)
        self.merge(also_test)
        bad = self.commit("bad")

        self.logger.info("\n" + self.git("log", "--graph", "--all", "--oneline"))

        self.assertEqual(git_dissect.dissect(f"{good}..{bad}", ["sh", "./run.sh"]), want)
        # All we do is find the first commit in the bisection range where things
        # went from good to bad (this is just how git-bisect works). It's
        # reasonable that we only tested exactly the two commits we needed to
        # achieve that.
        runs = self.script_runs()
        self.assertIn(want, runs)

    def _run_worktree_test(self, use_worktrees: bool, cleanup_worktrees=False):
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
        init = self.commit("init")
        for i in range(50):
            self.commit("good " + str(i))
        self.write_pass_fail_script(fail=True)
        end = self.commit("end")

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        git_dissect.dissect(f"{init}..{end}", ["sh", "./log_cwd.sh", "./run.sh"],
                            use_worktrees=use_worktrees, cleanup_worktrees=cleanup_worktrees,
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

    def test_worktree_mode(self):
        for (cwd, dirty) in self._run_worktree_test(use_worktrees=True,
                                                    cleanup_worktrees=True):
            self.assertNotEqual(cwd, os.getcwd())
            # This is not actually required, any directory is fine, just a hack
            # assertion to try and catch scripts getting run in totally random
            # trees.
            self.assertNotEqual(cwd.find("worktree-"), -1)
            self.assertFalse(dirty)

    def test_worktree_mode_no_cleanup(self):
        results = self._run_worktree_test(use_worktrees=True,
                                          cleanup_worktrees=False)
        # If this flakes, it might be that by chance no worktree got reused,
        # which isn't technically a bug. Try to find a way to force them to get
        # reused.
        self.assertTrue(any(dirty for (cwd, dirty) in results))

    def test_non_worktree_mode(self):
        results = self._run_worktree_test(use_worktrees=False)

        # If nobody saw a dirty tree, seems like there was more parallelism than
        # expected, or we were racily cleaning up the tree.
        self.assertTrue(any(dirty for (cwd, dirty) in results))
        self.assertTrue(all(cwd == os.getcwd() for (cwd, dirty) in results))

    def _test_thread_limit(self, num_threads):
        # At each commit will be a script that starts, touches a unique file
        # path, then hangs until a different unique filepath appers, then exits.
        commits = []
        for i in range(20):
            script = f"""
                touch {os.getcwd()}/started-{i}
                while ! [ -e {os.getcwd()}/stop-{i} ]; do
                    sleep 0.01
                done
                touch {os.getcwd()}/done-{i}
                exit {"1" if i > 17 else "0"}
            """
            self.write_executable("run.sh", script)
            commits.append(self.commit())

        self.logger.info("\n" + self.git("log", "--graph", "--all", "--oneline"))

        # Run the dissection in the background
        dissect_result = None
        def run_dissect():
            nonlocal dissect_result
            dissect_result = git_dissect.dissect(
                f"{commits[0]}..{commits[-1]}", args=["sh", "./run.sh"], num_threads=num_threads)

        thread = threading.Thread(target=run_dissect)
        thread.start()

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
            time.sleep(0.5)

            if not thread.is_alive():
                break

            # TODO: I think the failure here is a race condition in the thread
            # pool code. Probably instead of debugging, best to just clean it
            # up.
            self.assertEqual(len(tests_in_state("started")), i + num_threads,
                             "after stopping %d threads" % i)

            # Let one of the threads exit
            to_terminate = next(iter(running_tests()))
            open("stop-" + str(to_terminate), "w").close()

        self.logger.info("joining")
        thread.join()

        # If this fails then dissect() must have crashed.
        self.assertIsNotNone(dissect_result)

    def test_no_parallelism(self):
        self._test_thread_limit(1)

    def test_limited_threads(self):
        self._test_thread_limit(4)

    def _test_test_selection(self, num_good, num_bad, want_num_tests):
        # This runs a test to check that we're actually bisecting and not just
        # testing every commit or picking random commits or something.
        self.write_pass_fail_script(fail=False)
        commits = []
        for i in range(num_good):
            commits.append(self.commit(f"good {i}"))
        self.write_pass_fail_script(fail=True)
        for i in range(num_bad):
            commits.append(self.commit(f"bad {i}"))

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))
        result = git_dissect.dissect(f"{commits[0]}..{commits[-1]}",
                                     ["sh", "./run.sh"], num_threads=1)

        # Check we didn't cheat
        tested_commits = self.script_runs()
        self.assertEqual(result, commits[num_good])
        self.assertIn(commits[num_good - 1], tested_commits)
        self.assertIn(commits[num_good], tested_commits)

        self.assertEqual(len(tested_commits), want_num_tests)

    def test_test_selection_1(self):
        self._test_test_selection(10, 5, 4)

    def test_test_selection_2(self):
        self._test_test_selection(5, 10, 4)

    def test_test_selection_3(self):
        self._test_test_selection(2, 2, 2)

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

class TestTestEveryCommit(GitDissectTest):
    def test_smoke(self):
        self.write_pass_fail_script(fail=False)
        commits = []
        commits.append(self.commit())
        commits.append(self.commit())
        commits.append(self.commit())
        self.write_pass_fail_script(fail=True)
        commits.append(self.commit())
        commits.append(self.commit())
        commits.append(self.commit())
        self.write_pass_fail_script(fail=False)
        commits.append(self.commit())

        self.logger.info(self.git("log", "--graph", "--all", "--oneline"))

        results = git_dissect.test_every_commit(
            f"{commits[0]}..{commits[-1]}",
            ["sh", "./run.sh"])
        want = list(reversed(list(zip(commits[1:], [
            0, 0, 1, 1, 1, 0
        ]))))

        self.assertCountEqual(results, want)

class TestRevRange(GitDissectTest):
    def test_full_spec(self):
        # Hack: when specifying the range in full we don't actually need access
        # to the repo so we dont bother using "real" refs here.
        # This would fail if you tried to access "commits()" etc.

        r = git_dissect.RevRange.from_string("foo ^bar ^baz")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_dissect.RevRange.from_string("foo   ^bar") # multiple spaces
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar"])

        r = git_dissect.RevRange.from_string("^bar foo ^baz")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_dissect.RevRange.from_string("^bar ^baz foo")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, ["bar", "baz"])

        r = git_dissect.RevRange.from_string("foo")
        self.assertEqual(r.include, "foo")
        self.assertCountEqual(r.exclude, [])

        with self.assertRaises(git_dissect.BadRangeError):
            _ = git_dissect.RevRange.from_string("^foo")

        with self.assertRaises(git_dissect.BadRangeError):
            _ = git_dissect.RevRange.from_string("foo bar")

    def test_dot_dot(self):
        r = git_dissect.RevRange.from_string("foo..bar")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo"])

        r = git_dissect.RevRange.from_string("foo..bar ^baz")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo", "baz"])

        r = git_dissect.RevRange.from_string("^baz foo..bar")
        self.assertEqual(r.include, "bar")
        self.assertCountEqual(r.exclude, ["foo", "baz"])

        with self.assertRaises(git_dissect.BadRangeError):
            _ = git_dissect.RevRange.from_string("foo..bar..baz")

        with self.assertRaises(git_dissect.BadRangeError):
            _ = git_dissect.RevRange.from_string("foo..bar baz")


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
    lambda children: with_additional_edge(children) | with_additional_node(children))

# Strategy that generates a DAG and a pair of ("smaller", "larger") node IDs
# within that DAG.
@hypothesis.strategies.composite
def with_node_range(draw, dags):
    dag = draw(dags.filter(lambda d: d.num_nodes > 1))
    return (dag, draw(distinct_sorted_uint_pairs(max_value=dag.num_nodes-1)))

# Strategy that generates a DAG and the ID of a node within that DAG, and a leaf
# node that is reachable from that other node.
@hypothesis.strategies.composite
def with_node_and_reachable_leaf(draw, dags):
    dag = draw(dags)
    leaves = set(range(dag.num_nodes))
    for src, _ in dag.edges:
        if src in leaves:
            leaves.remove(src)
    leaf_id = sorted(leaves)[draw(uints_upto(len(leaves) - 1))]
    leaf_ancestors = dag.nodes()[leaf_id].ancestor_ids()
    ancestor_id = leaf_ancestors[draw(uints_upto(len(leaf_ancestors) - 1))]
    return (dag, ancestor_id, leaf_id)

class TestWithHypothesis(GitDissectTest):
    repo_cache: dict[Dag, str] = {}  # Maps DAGs to repo paths

    def setUp(self):
        self.logger = logging.getLogger(self.id())
        # Needed for inherited self.commit. TODO: Cleanup inheritance mess
        self.commit_counter = 0

    # Sets up and switches to a repo where the history structure matches the
    # DAG. Each commit is tagged with the corresponding node's ID.
    def setup_repo(self, dag):
        if path := self.repo_cache.get(dag):
            os.chdir(path)
            return

        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        self.git("init")

        commits = {}  # Maps DagNode.i to commit hash, faster than using git tags.
        def create_commit(node: DagNode):
            if node.i in commits:
                return

            if len(node.parents) == 0:
                # This is the "root" of the history. In this code we always have
                # exactly one of these - check that.
                self.assertEqual(len(commits), 0)

            for parent in node.parents:
                create_commit(parent)

            if node.parents:
                self.git("checkout", commits[node.parents[0].i])
            if len(node.parents) <= 1:
                commits[node.i] = self.commit(str(node.i))
            else:
                commits[node.i] = self.merge(*[commits[p.i] for p in node.parents])
            self.git("tag", str(node.i))

        for node in dag.nodes():
            create_commit(node)

        self.addClassCleanup(lambda: shutil.rmtree(tmpdir))

    def describe(self, rev: str) -> str:
        return self.git("describe", "--tags", rev).strip()

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

    @hypothesis.given(dag_and_range=with_node_range(dags()))
    @hypothesis.example((Dag(num_nodes=4, edges=frozenset({(0, 1), (0, 2), (2, 3), (1, 3)})),
                         (2, 3)))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=1))
    def test_range_split(self, dag_and_range: tuple[Dag, tuple[int, int]]):
        (dag, (exclude_node, include_node)) = dag_and_range
        self.setup_repo(dag)
        rev_range = git_dissect.RevRange(exclude=[str(exclude_node)],
                                         include=str(include_node))
        m = rev_range.midpoint()
        if not m:
            return # Range is empty
        before, after = rev_range.split(m)
        self.assertCommitSetPartition((before.commits(), after.commits()), rev_range.commits())

        # Same for dropping the tip of the range.
        subranges = rev_range.drop_include()
        self.assertCommitSetPartition((s.commits() for s in subranges),
                                      rev_range.commits() - {git_dissect.rev_parse(rev_range.include)})

    @hypothesis.given(dag_node_leaf=with_node_and_reachable_leaf(dags()))
    @hypothesis.example((Dag(num_nodes=4, edges=frozenset([(0, 1), (1, 2), (2, 3)])), 2, 3))
    @hypothesis.settings(deadline=datetime.timedelta(seconds=1))
    def test_bisect(self, dag_node_leaf: tuple[Dag, int, int]):
        dag, culprit_node_id, leaf_id = dag_node_leaf
        self.setup_repo(dag)
        # Bisect the range consisting of all the ancestors of the leaf node.
        range_spec = str(leaf_id)
        # Simulate a bug being introduced by our culprit commit. Comits are
        # tagged by the ID of the node they are generated from. We use that tag
        # to test if the culprit is an ancestor of HEAD; if it is then HEAD is
        # "broken".
        git_dissect.dissect(
            rev_range=range_spec,
            args=["git", "merge-base", "--is-ancestor", str(culprit_node_id), str(leaf_id)])

    # TODO: test multiple "good" that are not the root of the repo


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)-6s %(message)s")

    unittest.main()