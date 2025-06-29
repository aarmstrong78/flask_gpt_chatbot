# Guidelines for contributors

This repository hosts a simple Flask-based chatbot. Follow these conventions when making changes.

## Coding Style
- Format all Python code with `black` using a line length of 120.
- Write clear docstrings for new functions or classes.
- Keep imports sorted and remove unused imports.

## Required Checks
Before submitting changes run:

1. `black --check .`
2. `flake8`
3. `pytest -q`

All checks should pass.

## Pull Requests
Summarize your changes and reference relevant lines. Mention the outcome of the checks in the PR body.
