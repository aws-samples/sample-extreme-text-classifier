#!/bin/bash

# Text Classifier Demo Launch Script
# Compatible with Mac and Linux
# Respects local environment by using virtual environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# Function to find Python executable
find_python() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        # Check if it's Python 3
        if python --version 2>&1 | grep -q "Python 3"; then
            echo "python"
        else
            return 1
        fi
    else
        return 1
    fi
}

# Function to cleanup background processes on exit
cleanup() {
    print_status "Cleaning up background processes..."
    if [[ -n $BACKEND_PID ]]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [[ -n $FRONTEND_PID ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    # Deactivate virtual environment if active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Main script starts here
print_status "Starting Text Classifier Demo Setup..."

# Detect OS
OS=$(detect_os)
print_status "Detected OS: $OS"

# Check if we're in the right directory
if [[ ! -f "launch-demo.sh" ]]; then
    print_error "Please run this script from the ui/ directory"
    exit 1
fi

# Go to project root (parent directory)
cd ..

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
PYTHON_CMD=$(find_python)
if [[ $? -ne 0 ]]; then
    print_error "Python 3.8+ is required but not found"
    print_error "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_success "Found Python $PYTHON_VERSION"

# Check Node.js
if ! command_exists node; then
    print_error "Node.js is required but not found"
    print_error "Please install Node.js 16+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
print_success "Found Node.js $NODE_VERSION"

# Check npm
if ! command_exists npm; then
    print_error "npm is required but not found"
    print_error "Please install npm (usually comes with Node.js)"
    exit 1
fi

NPM_VERSION=$(npm --version)
print_success "Found npm $NPM_VERSION"

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."

if [[ ! -d ".venv" ]]; then
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Verify we're in the virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated: $VIRTUAL_ENV"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_success "Python dependencies installed"

# Setup Node.js dependencies
print_status "Setting up Node.js dependencies..."
cd ui/frontend

if [[ ! -d "node_modules" ]]; then
    print_status "Installing Node.js dependencies..."
    npm install > /dev/null 2>&1
    print_success "Node.js dependencies installed"
else
    print_status "Node.js dependencies already installed"
fi

# Go back to project root
cd ../..

# Start backend server
print_status "Starting backend server..."
cd ui/backend
$PYTHON_CMD start_backend.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    print_error "Backend server failed to start"
    print_error "Check ui/backend/backend.log for details"
    exit 1
fi

print_success "Backend server started (PID: $BACKEND_PID)"
print_status "Backend API available at: http://localhost:8000"
print_status "API documentation at: http://localhost:8000/docs"

# Start frontend server
print_status "Starting frontend development server..."
cd ui/frontend
npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

# Wait a moment for frontend to start
sleep 5

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend server failed to start"
    print_error "Check ui/frontend/frontend.log for details"
    exit 1
fi

print_success "Frontend server started (PID: $FRONTEND_PID)"
print_status "Frontend available at: http://localhost:3000"

# Try to open browser (if available)
print_status "Attempting to open demo in browser..."
if command_exists open && [[ "$OS" == "macos" ]]; then
    open http://localhost:3000
elif command_exists xdg-open && [[ "$OS" == "linux" ]]; then
    xdg-open http://localhost:3000
else
    print_warning "Could not auto-open browser. Please visit: http://localhost:3000"
fi

# Display status
echo ""
print_success "🚀 Text Classifier Demo is now running!"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
print_status "Press Ctrl+C to stop all servers"
echo ""

# Wait for user to stop
wait