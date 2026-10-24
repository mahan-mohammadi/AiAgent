#!/usr/bin/env bash
set -euo pipefail

# Run from the project root.
mkdir -p tools

# Tool modules
touch tools/__init__.py
touch tools/calculator.py
touch tools/datetime_tools.py
touch tools/text_tools.py
touch tools/unit_tools.py

# Agent architecture
touch graph.py
touch state.py
touch config.py

# Keep the existing entry point and project configuration.
touch main.py
touch pyproject.toml

echo "Project structure created:"
find . -maxdepth 2 -type f \
  ! -path './.venv/*' \
  ! -path './.git/*' |
  sort
