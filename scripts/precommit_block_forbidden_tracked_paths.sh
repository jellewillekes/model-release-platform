#!/usr/bin/env bash
set -euo pipefail

forbidden_regex='(^mlruns/|^mlartifacts/|^\.venv/|^venv/|^\.pytest_cache/|^\.mypy_cache/|^\.ruff_cache/|^__pycache__/|\.DS_Store$|^data/|^artifacts/)'

if git ls-files | egrep -n "${forbidden_regex}"; then
  echo ""
  echo "ERROR: Forbidden files are tracked by git."
  echo "Remove them from the index and add them to .gitignore."
  exit 1
fi
