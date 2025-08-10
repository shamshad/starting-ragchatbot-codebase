#!/bin/bash

# Run complete quality check workflow: format and then lint
set -e

echo "🚀 Running complete code quality workflow..."

# Format code first
./scripts/format.sh

# Then run all checks
./scripts/lint.sh

echo "✨ Quality workflow complete!"