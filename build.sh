#!/bin/bash
set -e

echo "Installing dependencies..."
if [ -f src/requirements.txt ]; then
    pip install -r src/requirements.txt
fi

echo "Validating app..."
if [ -f src/app.py ]; then
    python -m py_compile src/app.py
fi

echo "Build complete!"
