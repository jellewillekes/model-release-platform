#!/usr/bin/env bash
set -euo pipefail

forbidden_regex='(^mlruns/|^mlartifacts/|^\.venv/|^venv/|^\.pytest_cache/|^\.mypy_cache/|^\.ruff_cache/|^__pycache__/|\.DS_Store$|^data/|^artifacts/)'

if git ls-files | egrep -n "${forbidden_regex}"; then
  echo "Forbidden files are tracked by git. Remove from the index and add to .gitignore."
  exit 1
fi
