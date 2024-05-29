# List all commands.
default:
  @just --list

# Build docs.
docs:
  rm -rf docs/_build docs/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/_build/html/index.html

# Do a dev install.
dev:
  pip install -e '.[dev]'
  conda env update --file environment.yml

# Do a dev install with GPU support.
dev-gpu:
  pip install -e '.[dev-gpu]'
  conda env update --file environment.yml

# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff check src/ tests/ docs/ )

  echo
  ( set -x; ruff format --check src/ tests/ docs/ )

  echo
  ( set -x; mypy src/ tests/ docs/ )

  echo
  ( set -x; pytest )

  echo
  ( set -x; make -C docs doctest )

  test $error = 0

# Auto-fix code issues.
fix:
  ruff format src/ tests/ docs/
  ruff check --fix src/ tests/ docs/

# Build a release.
build:
  python -m build
