# Contributing to Polymetis
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
    - Trunk-based development: Branches should be short-lived and a PR should be submitted when a feature is complete.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
    - Run `pytest` and make sure it passes before pushing anything to `main` branch. Remember to have good test coverage for new features.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### Formatting

We enforce linters for our code. The `formatting` test will not pass if your code does not conform.

To make this easy for yourself, you can either
- Add the formattings to your IDE
- Install the git [pre-commit](https://pre-commit.com/) hooks by running
    ```bash
    pip install pre-commit
    pre-commit install
    ```

#### Python

We use [black](https://github.com/psf/black).

To enforce this in VSCode, install [black](https://github.com/psf/black), [set your Python formatter to black](https://code.visualstudio.com/docs/python/editing#_formatting) and [set Format On Save to true](https://code.visualstudio.com/updates/v1_6#_format-on-save).

To format manually, run: `black .`

#### C++

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html).

To automatically format in VSCode, install [clang-format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) and [set Format On Save to true](https://code.visualstudio.com/updates/v1_6#_format-on-save).

To format manually, run: `./scripts/format_cpp.sh format all`

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to Polymetis, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
