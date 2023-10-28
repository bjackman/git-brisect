# A Parallel Git Bisector

`git-dissect` does the same thing as `git bisect run`, but supports testing
multiple commits in parallel.

To install it, put `git_dissect.py` on your `$PATH`, make it executable, and
rename it to `git-dissect`. Now `git` has a `dissect` subcommand.

To use it, run `git dissect $bad_commit..$good_commit -- <test command>`.

## Writing the test command

The test command should exit with code 0 if the commit is "good", and a non-zero
code if it's bad. `git-dissect` will find you a _culprit_, i.e. a commit that is
"bad" whose parents are all "good". The good and bad commits specified in your
commandline are not tested, `git-dissect` trusts you that they are bad/good.
That means if _every_ commit is actually good, `git-dissect` will say that your
`$bad_commit` is the culprit.

By default, `git-dissect` will create a worktree for each thread (per
`--num-threads`). When a thread is idle, it will pick a commit that needs to be
tested, check it out in its worktree, and run your test command from in there.
No cleanup is performed between tests, if you need that then you should do it in
your test command.

When it is detected that a running test is no longer of interest (e.g. if the
test for the parent commit completes first and is determined to be bad), the
test is `SIGTERM`'d. Ideally it should shut down to free up the thread to test
another commit. Its exit code doesn't matter in that case.

The test command is not run via the shell, so you might want to use `bash -c
<commands>` or something.

If your tests produce other outputs that might be interesting to examine later,
the command can store them in `$GIT_DISSECT_OUTPUT_DIR` which is unique for each
commit (but remember, the command isn't run via the shell). You don't need to
create the directory, it already exists. Only files that you create go in here;
you don't need to worry about filename collisions.

## The result directory

`git-dissect` will collect the command's outputs and store it in a directory.
You can configure that directory with `--out-dir` and `--out-dir-in` (see `-h`
for more detail).

Each commit's test output is stored in a directory named after the full
commit hash.

 - If the test was aborted early, an empty file called `CANCELED` (with one
   "L") is there. Take note of this before reading too much into the other
   files.
 - `stderr.txt` and `stdout.txt` contain what you expect.
 - `returncode.txt` has the returncode as a decimal string.
 - `output/` contains anything your test command dropped into
   `$GIT_DISSECT_OUTPUT_DIR`.

## Discussion

`git-dissect` is similar to
[`git-pisect`](https://github.com/hoelzro/git-pisect/blob/master/git-pisect) but
has a more sophisticated algorithm. In particular, each thread will immediately
begin testing a new commit once the previous test is complete, while in
`git-pisect` all threads must complete their current test before any can begin
the next. This means `git-dissect` can be expected to perform dramatically
better when the time to complete tests of different commits varies, for example
when running stress-tests that detect a bug by repeating some operation millions
of times over the course of several minutes.


- why not a drop-in for `git-bisect`
- thoughts about the code
- missing features

##

`git-dissect` is not a drop-in replacement for `git-bissect`

## Running the tests

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# I wrote a bunch of shit code and then tried both of these two type checkers.
# They found different bugs. So I guess we're just gonna run both.
mypy *.py
pytype *.py
python3 git_dissect_test.py -c -f

# Or to just run a single test:
python3 git_dissect_test.py -c -f -b TestWithHypothesis.test_range_split

# Or if you want stats from Hypothesis:
pytest --hypothesis-show-statistics
pytest git_dissect_test.py::TestWithHypothesis::test_range_split --hypothesis-show-statistics
```