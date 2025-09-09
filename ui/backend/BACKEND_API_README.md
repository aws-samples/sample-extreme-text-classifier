# Multi-Class Text Classifier Backend API

This FastAPI backend server provides REST API endpoints for the multi-class text classifier frontend showcase application.

## Features

- **Dataset Management**: Create, read, update, and delete classification datasets
- **Text Classification**: Classify text documents with configurable options
- **PDF Classification**: Extract text from PDFs and classify them
- **Attribute Management**: Generate and manage class attributes for enhanced classification
- **Domain Wizard**: Generate complete datasets for specific domains
- **Example Generation**: Create sample documents for testing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
cd ui/backend
python start_backend.py
```

The server will start on `http://127.0.0.1:8000`

### 3. Access API Documentation

- **Interactive API Docs**: http://127.0.0.1:8000/docs
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc

### 4. Test the API

```bash
cd ui/backend
python test_backend_api.py
```

## API Endpoints

### Health Check

- `GET /health` - Check server health and service availability

### Dataset Management

- `GET /datasets` - List all available datasets
- `GET /datasets/{dataset_id}` - Get a specific dataset
- `POST /datasets` - Create a new dataset
- `PUT /datasets/{dataset_id}` - Update an existing dataset
- `DELETE /datasets/{dataset_id}` - Delete a dataset
- `POST /datasets/{dataset_id}/embeddings` - Generate embeddings for a dataset

### Classification

- `POST /classify/text` - Classify text content
- `POST /classify/pdf` - Upload and classify PDF documents

### Attribute Management

- `GET /attributes/{dataset_id}` - Get attributes for a dataset
- `POST /attributes/{dataset_id}/generate` - Generate attributes using LLM
- `PUT /attributes/{dataset_id}` - Save attributes for a dataset

### Domain Wizard

- `POST /wizard/generate-dataset` - Generate a complete dataset for a domain

### Example Generation

- `POST /examples/generate` - Generate example documents for a class

## Directory Structure

The API manages files in the following directories:

```
ui/
├── backend/           # Backend API server files
│   ├── backend_api.py
│   ├── start_backend.py
│   └── test_backend_api.py
└── frontend/          # Frontend application and data
    ├── datasets/      # Dataset JSON files and embeddings
    ├── output/        # Generated outputs and results
    └── temp/          # Temporary files (auto-cleaned)
```

## Sample Datasets

The API comes with two sample datasets:

1. **Sample Documents** (`sample_documents`) - Common document types (invoices, contracts, resumes, etc.)
2. **Medical Supplies** (`medical_supplies`) - Medical equipment and supplies classification

## Configuration Options

### Classification Configuration

```json
{
  "use_reranking": false,
  "reranking_model": "us.amazon.nova-lite-v1:0",
  "use_attribute_validation": false,
  "top_k_candidates": 10
}
```

### Dataset Model

```json
{
  "id": "dataset_id",
  "name": "Dataset Name",
  "description": "Dataset description",
  "classes": [
    {
      "name": "Class Name",
      "description": "Class description",
      "examples": ["example 1", "example 2"],
      "metadata": {}
    }
  ],
  "embeddings_generated": false,
  "created_at": "2024-01-15T10:00:00",
  "updated_at": "2024-01-15T10:00:00"
}
```

## Error Handling

The API returns structured error responses:

```json
{
  "error": "Error message",
  "details": "Additional error details",
  "type": "error_type"
}
```

Error types:
- `validation_error` - Invalid input data
- `configuration_error` - Service configuration issues
- `processing_error` - Classification or processing failures
- `classifier_error` - Library-specific errors
- `internal_error` - Generic server errors

## Development

### Running in Development Mode

The server runs with auto-reload enabled by default. Changes to the code will automatically restart the server.

```bash
cd ui/backend
python start_backend.py
```

### Adding New Endpoints

1. Add the endpoint function to `backend_api.py`
2. Define request/response models using Pydantic
3. Add proper error handling using `handle_api_error()`
4. Update this documentation

### Testing

Use the provided test script to verify functionality:

```bash
cd ui/backend
python test_backend_api.py
```

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:3000` (React development server)
- `http://127.0.0.1:3000`

## Security Considerations

This is a development/demo server with basic security:
- No authentication required
- File operations restricted to `frontend/` directory
- Input validation on all endpoints
- Error messages sanitized to avoid information disclosure

For production use, consider adding:
- Authentication and authorization
- Rate limiting
- Input sanitization
- HTTPS support
- Logging and monitoring

## Troubleshooting

### Server Won't Start

1. Check if port 8000 is available
2. Verify all dependencies are installed
3. Check Python version compatibility (3.8+)

### Classification Errors

1. Ensure embeddings are generated for the dataset
2. Check that the dataset has valid class definitions
3. Verify the classification library is properly installed

### File Operation Errors

1. Check directory permissions
2. Ensure sufficient disk space
3. Verify the `frontend/` directory structure exists

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Run the test suite to identify problems: `cd ui/backend && python test_backend_api.py`
3. Check server logs for detailed error information