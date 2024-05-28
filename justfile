# List all commands.
default:
  @just --list

# Build docs.
docs:
  rm -rf docs/build docs/source/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/build/html/index.html

conda-deps := " \
  conda-forge::ipopt \
  conda-forge::pyscipopt \
  conda-forge::coin-or-cbc \
"

# Do a dev install.
dev:
  pip install -e '.[dev]'
  conda install {{conda-deps}}

# Do a dev install with GPU support.
dev-gpu:
  pip install -e '.[dev-gpu]'
  conda install {{conda-deps}}

# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff check src/ tests/ docs/source/ examples/ )

  echo
  ( set -x; ruff format --check src/ tests/ docs/source/ examples/ )

  echo
  ( set -x; mypy src/ tests/ docs/source/ examples/ )

  echo
  ( set -x; pytest )

  echo
  ( set -x; make -C docs doctest )

  test $error = 0

# Auto-fix code issues.
fix:
  ruff format src/ tests/ docs/source/ examples/
  ruff check --fix src/ tests/ docs/source/ examples/

# Build a release.
build:
  python -m build
