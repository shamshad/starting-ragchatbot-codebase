#!/bin/bash

# Format Python code with black and sort imports with isort
set -e

echo "🔧 Formatting Python code..."

echo "  → Running black..."
uv run black backend/ main.py

echo "  → Sorting imports..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"