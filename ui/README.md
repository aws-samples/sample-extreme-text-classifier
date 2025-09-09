# Text Classifier Frontend Demo

This is a comprehensive frontend demo application that showcases the capabilities of the multi-class text classifier library. The demo provides an intuitive interface for exploring classification features, dataset management, and document processing capabilities.

## Architecture

The demo consists of two main components:

- **Frontend**: React TypeScript application with Tailwind CSS
- **Backend**: FastAPI Python server that interfaces with the text classifier library

## Features

- **Interactive Wizard**: Step-by-step interface for generating classification attributes
- **Dashboard**: Overview of classification results and dataset management
- **Document Processing**: Upload and classify PDF documents
- **Real-time Classification**: Live classification results with confidence scores
- **Dataset Management**: View and manage classification datasets

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Virtual environment support** (recommended)
- **AWS credentials configured** with AWS CLI

## Quick Start

### Option 1: Using the Launch Script (Recommended)

The easiest way to get started is using our cross-platform launch script:

```bash
# Make the script executable (Mac/Linux)
chmod +x launch-demo.sh

# Launch both frontend and backend
./launch-demo.sh
```

The script will:
- Create and activate a Python virtual environment
- Install Python dependencies
- Install Node.js dependencies
- Start the backend server (port 8000)
- Start the frontend development server (port 3000)
- Open the demo in your default browser

### Option 2: Manual Setup

If you prefer to set up manually or need more control:

#### Backend Setup

1. **Create and activate virtual environment**:
   ```bash
   # From the project root directory
   python -m venv .venv
   
   # Activate virtual environment
   # On Mac/Linux:
   source .venv/bin/activate
   # On Windows:
   # .venv\Scripts\activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**:
   ```bash
   cd ui/backend
   python start_backend.py
   ```

   The backend API will be available at `http://localhost:8000`

#### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   cd ui/frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## Usage

1. **Access the Demo**: Open `http://localhost:3000` in your browser
2. **Explore the Wizard**: Use the step-by-step wizard to generate classification attributes
3. **Upload Documents**: Test document classification with PDF files
4. **View Results**: Check the dashboard for classification results and insights

## API Documentation

The backend API documentation is available at `http://localhost:8000/docs` when the server is running.

## Development

### Frontend Development

The frontend is built with:
- **React 19** with TypeScript
- **Tailwind CSS** for styling
- **Zustand** for state management
- **React Router** for navigation
- **Lucide React** for icons

Key directories:
- `src/components/` - Reusable UI components
- `src/pages/` - Main application pages
- `src/stores/` - State management
- `src/types/` - TypeScript type definitions

### Backend Development

The backend uses:
- **FastAPI** for the web framework
- **Uvicorn** as the ASGI server
- **Pydantic** for data validation
- Integration with the main text classifier library

Key files:
- `backend_api.py` - Main API endpoints
- `start_backend.py` - Server startup script
- `services/` - Business logic and integrations

## Troubleshooting

### Common Issues

1. **Port conflicts**: If ports 3000 or 8000 are in use, modify the launch script or start servers manually on different ports

2. **Python virtual environment**: Ensure you're using the virtual environment:
   ```bash
   which python  # Should point to .venv/bin/python
   ```

3. **Node.js dependencies**: If you encounter npm issues, try:
   ```bash
   cd ui/frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Backend connection**: The frontend expects the backend at `http://localhost:8000`. If running on different ports, update the API base URL in the frontend configuration.

### Logs and Debugging

- **Backend logs**: Check the terminal where you started the backend server
- **Frontend logs**: Check the browser developer console
- **API testing**: Use the interactive docs at `http://localhost:8000/docs`

## Contributing

When making changes to the demo:

1. Follow the coding standards defined in the project
2. Test both frontend and backend components
3. Update documentation as needed
4. Ensure the launch script continues to work on both Mac and Linux

## License

This demo application is part of the multi-class text classifier library project.