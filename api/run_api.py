#!/usr/bin/env python3
"""
Startup script for the Model Merging API

Usage:
    python run_api.py

This script starts the Flask API server for model merging operations.
"""
import os
import sys

# Add the parent directory to Python path to import merging-eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ['TRANSFORMERS_NO_TORCHVISION'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

if __name__ == '__main__':
    from app import app

    print("=" * 60)
    print("Model Merging API Server")
    print("=" * 60)
    print("Starting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("API methods: http://localhost:5000/api/methods")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )