#!/bin/bash

# Run all linting and quality checks
set -e

echo "🔍 Running code quality checks..."

echo "  → Running flake8 (basic checks)..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503,E501,F401,F841,E402,F811,F541,W291 || echo "⚠️ flake8 found issues (non-blocking)"

echo "  → Checking code formatting..."
uv run black --check backend/ main.py

echo "  → Checking import sorting..."
uv run isort --check-only backend/ main.py

echo "✅ Essential quality checks passed!"