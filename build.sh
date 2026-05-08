#!/bin/bash
# Build script for dxo-client package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building dxo-client ==="

# Clean previous builds (ignore errors for permission issues)
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info dxo_client.egg-info 2>/dev/null || true

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Install build tool
echo "Installing build tool..."
.venv/bin/pip install --upgrade pip build -q

# Build the package
echo "Building package..."
.venv/bin/python -m build

echo ""
echo "=== Build complete ==="
echo "Output files:"
ls -la dist/

echo ""
echo "To install locally:"
echo "  pip install dist/dxo_client-*.whl"
echo ""
echo "To install in another project:"
echo "  pip install /path/to/dxo-client/dist/dxo_client-*.whl"
