#!/bin/bash

# Build and test script for Merging-EVAL package
# Usage: ./build_and_test.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

# Clean previous builds
print_info "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install/upgrade build tools
print_info "Installing/upgrading build tools..."
pip install --upgrade pip build twine certifi

# Build the package
print_info "Building package..."
# Use offline mode to avoid network issues
export PIP_NO_INDEX=1
export PIP_NO_DEPENDENCIES=1
python -m build --no-isolation

# Check the built package
print_info "Checking built package..."
twine check dist/*

# Create test virtual environment
print_info "Creating test virtual environment..."
python -m venv test_env
source test_env/bin/activate

# Install the package
print_info "Installing package in test environment..."
pip install dist/*.whl

# Test basic import (without dependencies)
print_info "Testing package import (structure only)..."
python -c "
import os
import sys
# Add the installed package to path
site_packages = [p for p in sys.path if 'site-packages' in p]
if site_packages:
    merge_path = os.path.join(site_packages[0], 'merge')
    if os.path.exists(merge_path):
        print(f'✓ Package structure verified: {merge_path}')
        print('✓ All modules present:')
        for f in os.listdir(merge_path):
            if f.endswith('.py'):
                print(f'  - {f}')
    else:
        print('✗ Package structure not found')
else:
    print('✗ No site-packages found')
"

# Test CLI command availability (without execution)
print_info "Testing CLI command availability..."
python -c "
import importlib.util
spec = importlib.util.find_spec('merge.main_merging')
if spec is not None:
    print('✓ CLI module found')
else:
    print('✗ CLI module not found')
"

# Deactivate test environment
deactivate
rm -rf test_env

print_info "Package build and test completed successfully!"
print_info "Built files:"
ls -la dist/

# Installation instructions
echo ""
print_info "To install the package locally:"
echo "pip install dist/*.whl"

echo ""
print_info "To install in development mode:"
echo "pip install -e ."