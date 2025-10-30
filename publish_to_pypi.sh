#!/bin/bash

# Publishing script for Merging-EVAL to PyPI mirror
# Usage: ./publish_to_pypi.sh [test|production]

set -e

# Configuration
PACKAGE_NAME="merging-eval"
PYPI_TEST_URL="https://test.pypi.org/legacy/"
PYPI_PROD_URL="https://upload.pypi.org/legacy/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Determine target (test or production)
TARGET="${1:-test}"

case "$TARGET" in
    "test")
        PYPI_URL="$PYPI_TEST_URL"
        print_info "Publishing to TestPyPI..."
        ;;
    "production")
        PYPI_URL="$PYPI_PROD_URL"
        print_warning "Publishing to production PyPI..."
        ;;
    *)
        print_error "Invalid target: $TARGET. Use 'test' or 'production'."
        exit 1
        ;;
esac

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

# Upload to PyPI
print_info "Uploading to $TARGET PyPI..."
if [ "$TARGET" = "test" ]; then
    twine upload --repository-url "$PYPI_URL" dist/*
else
    twine upload dist/*
fi

print_info "Package successfully published to $TARGET PyPI!"
print_info "Package name: $PACKAGE_NAME"
print_info "Files uploaded:"
ls -la dist/

# Installation instructions
if [ "$TARGET" = "test" ]; then
    echo ""
    print_info "To install from TestPyPI:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ $PACKAGE_NAME"
else
    echo ""
    print_info "To install from PyPI:"
    echo "pip install $PACKAGE_NAME"
fi