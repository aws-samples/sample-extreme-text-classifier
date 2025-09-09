#!/usr/bin/env python3
"""
Startup script for the FastAPI backend server.

This script ensures all dependencies are available and starts the server
with appropriate configuration for the frontend showcase.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        Path("frontend/datasets"),
        Path("frontend/output"),
        Path("frontend/temp")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")


def start_server():
    """Start the FastAPI server."""
    print("Starting Multi-Class Text Classifier API server...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("API documentation will be available at: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Import and run the server
        import uvicorn
        uvicorn.run(
            "backend_api:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def main():
    """Main function."""
    print("Multi-Class Text Classifier Backend Server")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories()
    
    # Start server
    start_server()


if __name__ == "__main__":
    main()