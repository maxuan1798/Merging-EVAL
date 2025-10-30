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
pip install --upgrade pip build twine

# Build the package
print_info "Building package..."
python -m build

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

# Test basic import
print_info "Testing package import..."
python -c "import merge; import expert_training; print('Package imports successfully!')"

# Test CLI command (if available)
print_info "Testing CLI command..."
python -c "from merge.main_merging import main; print('CLI function available!')"

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