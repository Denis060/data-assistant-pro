#!/bin/bash

# Data Assistant Pro - Environment Setup Script
# This script sets up the development environment for the Data Assistant Pro project

set -e  # Exit on any error

echo "🚀 Setting up Data Assistant Pro development environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📍 Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
else
    echo "✅ Virtual environment already active: $VIRTUAL_ENV"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing production dependencies..."
pip install -r requirements.txt

echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Run initial checks
echo "🧪 Running initial code quality checks..."
black --check . || (echo "⚠️  Code formatting issues found. Run 'make format' to fix them.")
isort --check-only . || (echo "⚠️  Import sorting issues found. Run 'make format' to fix them.")

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import streamlit
import pandas
import numpy
import plotly
import sklearn
print('✅ All core dependencies imported successfully')
"

# Create necessary directories
mkdir -p logs
mkdir -p data/uploads

echo "
🎉 Setup complete! 

Next steps:
1. Start the application: make run
2. Run tests: make test
3. Format code: make format
4. Check all quality tools: make ci-check

Happy coding! 🚀
"
