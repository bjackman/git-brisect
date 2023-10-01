Some shitty Python hacks that are gradually evolving into a parallel Git bisector.

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