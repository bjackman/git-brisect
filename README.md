# A Parallel Git Bisector

`git-brisect` does the same thing as `git bisect run`, but supports testing
multiple commits in parallel.

To install it, just download `git-brisect` and put it on your `$PATH`.
Now `git` has a `brisect` subcommand.

To use it, run `git brisect $bad_commit..$good_commit -- <test command>`.

I expect it to work on any Unix, and Windows _might_ work (let me know if you
try it!).

This finds you a commit that is "bad", and whose parent commits are all "good".
You don't need to have a linear history, this handles merge commits just fine.
(`git bisect` does this too). You also don't have to specify a singular "good"
commit, you can specify `bad ^good1 ^good2` (you can only have one "bad" commit
though).

## Writing the test command

The test command should exit with code 0 if the commit is "good", and a non-zero
code if it's bad. `git-brisect` will find you a _culprit_, i.e. a commit that is
"bad" whose parents are all "good". The good and bad commits specified in your
commandline are not tested, `git-brisect` trusts you that they are bad/good.
That means if _every_ commit is actually good, `git-brisect` will say that your
`$bad_commit` is the culprit.

By default, `git-brisect` will create a worktree for each thread (per
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
the command can store them in `$GIT_BRISECT_OUTPUT_DIR` which is unique for each
commit (but remember, the command isn't run via the shell). You don't need to
create the directory, it already exists. Only files that you create go in here;
you don't need to worry about filename collisions.

## The result directory

`git-brisect` will collect the command's outputs and store it in a directory.
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
   `$GIT_BRISECT_OUTPUT_DIR`.

## Discussion

`git-brisect` is similar to
[`git-pisect`](https://github.com/hoelzro/git-pisect/blob/master/git-pisect) but
has a more sophisticated algorithm. In particular, each thread will immediately
begin testing a new commit once the previous test is complete, while in
`git-pisect` all threads must complete their current test before any can begin
the next. This means `git-brisect` can be expected to perform dramatically
better when the time to complete tests of different commits varies, for example
when running stress-tests that detect a bug by repeating some operation millions
of times over the course of several minutes.

It also has more advanced logic to select which commits to prioritize testing
for; it ought to do a better job htan `git-pisect` on nonlinear histories,
although this is actually quite a tricky problem and it's still far from
optimal. See the code comments if you're interested!

It was originally called `git-dissect` but then I realised [that already
exists](https://github.com/talshorer/git-dissect)! That is instead focussed on
distributing the testing across multiple hosts. You could also use `git-brisect`
to do that if you wanted to, you'd just need to do some more scripting of your
own. If you're interested in that, see the `--help` for the `--no-worktrees`
option.

`git-brisect` is not a drop-in replacement for `git-bisect`: Instead of
specifying a range for `git bisect` it is hard coded as the difference between a
set of `refs/bisect/good-*` and a `refs/bisect/bad`. This allows the logic to be
executed step by step across multiple command invocations. That's a helpful
design for the interactive nature of `git-bisect`, but `git-brisect` isn't
interactive so instead we just pass a range. This allows `git-brisect` to leave
the repository totally untouched (aside from creating worktrees). You could even
run a normal `git-bisect` in your repo while `git-brisect` is running in the
background.

## Shortcomings & Missing Features

 - Want an equivalent to `git bisect skip`.
 - Want the ability to re-use existing worktrees instead of creating and tearing down
   special ones.
 - Want to cache results based on the tree hash: most tests probably don't need
   to be repeated if the actual code is the same. This would need to be enabled
   by a flag.
 - Would be helpful to have a flag like `--test-good` where instead of trusting
   the user we also test the tip commit. It would be prioritized for initial
   testing, this would help catch the case that there's something wrong with the
   test command, or the user is mistaken about the test range.
 - The error handling is overall pretty ropey. I'm sure users will get hit with
   backtraces. Also the multithreading has no way to propagate unexpected errors
   back to the main thread so it's probably possible to make this thing hang.
 - Want a way to spin up a pipe that can be passed in to `make`'s
   `--jobserver-fd` thing.  - I think the test coverage of the actual algorithm
   is pretty good (I used [Hypothesis](https://hypothesis.readthedocs.io). But
   not for the end-to-end logic, like it's probably possible to confuse this
   thing with invalid inputs. Also the test code is quite a mess, but whatever.
 - There's some dead code in there for an incomplete `--test-every-commit`
   feature, I think I will probably still want to add that at some point.

## Running the tests

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytype *.py
python3 git_brisect_test.py -c -f

# Or to just run a single test:
python3 git_brisect_test.py -c -f -b TestWithHypothesis.test_range_split

# Or if you want stats from Hypothesis:
pytest --hypothesis-show-statistics
pytest git_brisect_test.py::TestWithHypothesis::test_range_split --hypothesis-show-statistics
```