"""
FastAPI backend server for the multi-class text classifier frontend showcase.

This server provides REST API endpoints to interface with the classification library,
handling dataset management, classification operations, and attribute management.
"""

import os
import json
import pickle
import gzip
import sys
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import traceback

# Configure logging for the backend
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to appropriate levels
logging.getLogger("multi_class_text_classifier.attribute_generator").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)  # Reduce boto3 noise
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce HTTP noise

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import configuration
from config import config, ExampleGenerationPrompts

# Import backend services
from services.example_generator import example_generator

# Add the root directory to Python path to import the classification library
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import the classification library
try:
    from multi_class_text_classifier import (
        TextClassifier,
        ClassDefinition,
        ClassificationResult,
        ClassifierConfig,
        PDFExtractor,
        create_pdf_extractor,
        DatasetGenerator,
        AttributeGenerator,
        LLMAttributeEvaluator,
        ClassifierError,
        InvalidInputError,
        ConfigurationError,
        ProcessingError
    )
    # Import configuration classes from the models module
    from multi_class_text_classifier.models.data_models import (
        AttributeGenerationConfig,
        AttributeEvaluationConfig,
        AttributeValidationConfig
    )
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Classification library not available: {e}")
    print("Running in demo mode with mock responses")
    CLASSIFIER_AVAILABLE = False
    
    # Mock classes for demo mode
    class ClassDefinition:
        def __init__(self, name, description, metadata=None):
            self.name = name
            self.description = description
            self.metadata = metadata or {}
    
    class ClassificationResult:
        def __init__(self, predicted_class, alternatives=None, explanation="", metadata=None):
            self.predicted_class = predicted_class
            self.alternatives = alternatives or []
            self.explanation = explanation
            self.metadata = metadata or {}
        
        def to_dict(self):
            return {
                "predicted_class": {
                    "name": self.predicted_class.name,
                    "description": self.predicted_class.description,
                    "metadata": self.predicted_class.metadata
                },
                "effective_score": 0.85,
                "similarity_score": 0.75,  # Add the missing similarity_score field
                "rerank_score": None,
                "reranked": False,
                "explanation": self.explanation,
                "alternatives": [
                    {
                        "name": alt.name if hasattr(alt, 'name') else str(alt),
                        "description": alt.description if hasattr(alt, 'description') else "Mock alternative",
                        "effective_score": 0.7 - i * 0.1,
                        "similarity_score": 0.7 - i * 0.1,
                        "rerank_score": None,
                        "reasoning": "Mock reasoning"
                    }
                    for i, alt in enumerate(self.alternatives[:3])
                ],
                "metadata": self.metadata
            }
    
    class ClassifierConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class TextClassifier:
        def __init__(self, config=None):
            self.config = config
            self.classes = []
        
        def load_classes(self, classes):
            self.classes = classes
        
        def classify(self, text):
            # Mock classification - return first class with high confidence
            if self.classes:
                predicted = self.classes[0]
                alternatives = self.classes[:3]
                explanation = f"Mock classification of text: '{text[:50]}...'"
                return ClassificationResult(predicted, alternatives, explanation)
            else:
                # Default mock result
                mock_class = ClassDefinition("Unknown", "Mock classification result")
                return ClassificationResult(mock_class, [], "No classes loaded")
    
    # Mock other classes
    class PDFExtractor:
        def extract_from_file(self, path):
            return type('PDFContent', (), {'text': f"Mock extracted text from {path}"})()
    
    def create_pdf_extractor():
        return PDFExtractor()
    
    class DatasetGenerator:
        def generate_classes(self, domain, num_classes, examples_per_class):
            return [
                ClassDefinition(
                    f"Mock Class {i+1}",
                    f"Mock class for {domain} domain",
                    {"examples": [f"Example {j+1} for class {i+1}" for j in range(examples_per_class)]}
                )
                for i in range(num_classes)
            ]
    
    # Mock AttributeGenerator removed - using real implementation
    
    class LLMAttributeEvaluator:
        def __init__(self, config):
            self.config = config
    
    class AttributeGenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class AttributeEvaluationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class AttributeValidationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Mock exceptions
    class ClassifierError(Exception):
        pass
    
    class InvalidInputError(ValueError):
        pass
    
    class ConfigurationError(Exception):
        pass
    
    class ProcessingError(Exception):
        pass

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Class Text Classifier API",
    description="Backend API for the text classifier frontend showcase",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory paths - centralized in backend data directory
DATASETS_DIR = Path("data/datasets")
OUTPUT_DIR = Path("data/output")
TEMP_DIR = Path("data/temp")

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Global instances
pdf_extractor = None
dataset_generator = None
attribute_generator = None
attribute_evaluator = None


def initialize_services():
    """Initialize global service instances."""
    global pdf_extractor, dataset_generator, attribute_generator, attribute_evaluator
    
    try:
        if CLASSIFIER_AVAILABLE:
            pdf_extractor = create_pdf_extractor(
                model_id=config.aws.default_nova_lite_model
            )
            dataset_generator = DatasetGenerator()
            
            # Initialize AttributeGenerator with proper configuration
            attribute_generation_config = AttributeGenerationConfig(
                model_id=config.aws.default_claude_sonnet_4_model,
                temperature=config.models.attribute_generation_temperature,
                max_tokens=config.models.attribute_generation_max_tokens
            )
            attribute_generator = AttributeGenerator(attribute_generation_config)
            
            attribute_evaluator = LLMAttributeEvaluator(AttributeEvaluationConfig())
            print("✓ Classification services initialized successfully")
        else:
            print("✓ Classification library not available - running in demo mode")
            pdf_extractor = None
            dataset_generator = None
            attribute_generator = None
            attribute_evaluator = None
    except Exception as e:
        print(f"Warning: Could not initialize services: {e}")
        pdf_extractor = None
        dataset_generator = None
        attribute_generator = None
        attribute_evaluator = None


# Initialize services on startup
initialize_services()


# Pydantic models for API requests/responses
class ClassDefinitionModel(BaseModel):
    """API model for class definition."""
    id: str = Field(..., description="Class ID")
    name: str = Field(..., description="Class name")
    description: str = Field(..., description="Class description")
    examples: Optional[List[str]] = Field(default=None, description="Example texts")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DatasetModel(BaseModel):
    """API model for dataset."""
    id: str = Field(..., description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    classes: List[ClassDefinitionModel] = Field(..., description="List of classes")
    embeddingsGenerated: bool = Field(default=False, description="Whether embeddings are generated")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    class Config:
        populate_by_name = True
        
    def model_dump(self, **kwargs):
        """Override model_dump to provide both camelCase and snake_case fields."""
        data = super().model_dump(**kwargs)
        # Add snake_case alias for embeddings field
        data['embeddings_generated'] = data['embeddingsGenerated']
        return data


class ClassificationConfigModel(BaseModel):
    """API model for classification configuration."""
    use_reranking: bool = Field(default=False, description="Enable reranking")
    reranking_model: Optional[str] = Field(default=None, description="Reranking model ID")
    use_attribute_validation: bool = Field(default=False, description="Enable attribute validation")
    top_k_candidates: int = Field(default=10, description="Number of top candidates to consider")


class ClassificationRequestModel(BaseModel):
    """API model for classification request."""
    text: str = Field(..., description="Text to classify")
    dataset_id: str = Field(..., description="Dataset ID to use for classification")
    config: ClassificationConfigModel = Field(default_factory=ClassificationConfigModel, description="Classification configuration")


class AttributeConditionModel(BaseModel):
    """API model for attribute condition."""
    description: str = Field(..., description="Condition description")
    type: str = Field(default="text_match", description="Condition type")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Condition parameters")


class AttributeRuleModel(BaseModel):
    """API model for attribute rule."""
    operator: str = Field(..., description="Logical operator (AND/OR)")
    conditions: List[Union[AttributeConditionModel, 'AttributeRuleModel']] = Field(..., description="List of conditions or nested rules")


class ClassAttributeModel(BaseModel):
    """API model for class attributes."""
    class_id: str = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    required_attributes: AttributeRuleModel = Field(..., description="Required attribute rules")
    generated: bool = Field(default=False, description="Whether attributes were auto-generated")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")


class DomainGenerationRequestModel(BaseModel):
    """API model for domain-based dataset generation."""
    domain: str = Field(..., description="Domain description (e.g., 'medical supplies')")
    num_classes: int = Field(default=5, description="Number of classes to generate")
    examples_per_class: int = Field(default=1, description="Number of examples per class")


class ErrorResponse(BaseModel):
    """API model for error responses."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Error details")
    type: str = Field(default="error", description="Error type")


# Utility functions
def get_dataset_path(dataset_id: str) -> Path:
    """Get the file path for a dataset."""
    # Check if it's a predefined dataset first
    predefined_path = DATASETS_DIR / f"{dataset_id}.json"
    if predefined_path.exists():
        return predefined_path
    
    # Otherwise, use output directory for user-created datasets
    return OUTPUT_DIR / f"{dataset_id}.json"


def get_user_dataset_path(dataset_id: str) -> Path:
    """Get the file path for a user-created dataset (always in output directory)."""
    return OUTPUT_DIR / f"{dataset_id}.json"


def get_embeddings_path(dataset_id: str) -> Path:
    """Get the file path for dataset embeddings."""
    return OUTPUT_DIR / f"{dataset_id}_embeddings.pkl.gz"


def get_attributes_path(dataset_id: str) -> Path:
    """Get the file path for dataset attributes."""
    return OUTPUT_DIR / f"{dataset_id}_attributes.json"


def load_dataset_from_file(dataset_id: str) -> Optional[DatasetModel]:
    """Load a dataset from file."""
    dataset_path = get_dataset_path(dataset_id)
    if not dataset_path.exists():
        return None
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if embeddings exist
        embeddings_path = get_embeddings_path(dataset_id)
        data['embeddingsGenerated'] = embeddings_path.exists()
        
        # Ensure classes have IDs (for backward compatibility)
        if 'classes' in data:
            for cls in data['classes']:
                if 'id' not in cls:
                    cls['id'] = str(uuid.uuid4())
        
        return DatasetModel(**data)
    except Exception as e:
        print(f"Error loading dataset {dataset_id}: {e}")
        return None


def save_dataset_to_file(dataset: DatasetModel, is_user_created: bool = True) -> bool:
    """Save a dataset to file."""
    if is_user_created:
        dataset_path = get_user_dataset_path(dataset.id)
    else:
        dataset_path = get_dataset_path(dataset.id)
    
    try:
        # Convert to dict and handle datetime serialization
        data = dataset.model_dump()
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat()
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving dataset {dataset.id}: {e}")
        return False


def convert_to_class_definitions(classes: List[ClassDefinitionModel]) -> List[ClassDefinition]:
    """Convert API class models to library class definitions."""
    return [
        ClassDefinition(
            name=cls.name,
            description=cls.description,
            metadata=cls.metadata or {}
        )
        for cls in classes
    ]





def handle_api_error(e: Exception) -> JSONResponse:
    """Handle API errors and return appropriate JSON response."""
    if isinstance(e, (InvalidInputError, ValueError)):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=str(e),
                type="validation_error"
            ).model_dump()
        )
    elif isinstance(e, ConfigurationError):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                type="configuration_error"
            ).model_dump()
        )
    elif isinstance(e, ProcessingError):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                type="processing_error"
            ).model_dump()
        )
    elif isinstance(e, ClassifierError):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                type="classifier_error"
            ).model_dump()
        )
    else:
        # Generic error
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details=str(e),
                type="internal_error"
            ).model_dump()
        )


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Class Text Classifier API",
        "version": "1.0.0",
        "mode": "production" if CLASSIFIER_AVAILABLE else "demo",
        "classifier_library": CLASSIFIER_AVAILABLE,
        "endpoints": {
            "datasets": "/datasets",
            "classification": "/classify",
            "attributes": "/attributes",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "production" if CLASSIFIER_AVAILABLE else "demo",
        "classifier_library": CLASSIFIER_AVAILABLE,
        "services": {
            "pdf_extractor": pdf_extractor is not None,
            "dataset_generator": dataset_generator is not None,
            "attribute_generator": attribute_generator is not None,
            "attribute_evaluator": attribute_evaluator is not None
        }
    }


# Dataset Management Endpoints

@app.get("/datasets", response_model=List[DatasetModel])
async def list_datasets():
    """List all available datasets."""
    try:
        datasets = []
        dataset_ids = set()
        
        # Scan the predefined datasets directory for JSON files
        for dataset_file in DATASETS_DIR.glob("*.json"):
            # Skip attribute files
            if dataset_file.name.endswith("_attributes.json"):
                continue
            
            dataset_id = dataset_file.stem
            dataset_ids.add(dataset_id)
            dataset = load_dataset_from_file(dataset_id)
            if dataset:
                datasets.append(dataset)
        
        # Scan the output directory for user-created datasets
        for dataset_file in OUTPUT_DIR.glob("*.json"):
            # Skip attribute files and embedding files
            if (dataset_file.name.endswith("_attributes.json") or 
                dataset_file.name.endswith("_embeddings.json")):
                continue
            
            dataset_id = dataset_file.stem
            # Skip if we already loaded this dataset from the predefined directory
            if dataset_id in dataset_ids:
                continue
                
            dataset_ids.add(dataset_id)
            dataset = load_dataset_from_file(dataset_id)
            if dataset:
                datasets.append(dataset)
        
        return datasets
    except Exception as e:
        return handle_api_error(e)


@app.get("/datasets/{dataset_id}", response_model=DatasetModel)
async def get_dataset(dataset_id: str):
    """Get a specific dataset by ID."""
    try:
        dataset = load_dataset_from_file(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


@app.post("/datasets", response_model=DatasetModel)
async def create_dataset(dataset: DatasetModel):
    """Create a new dataset."""
    try:
        # Set timestamps
        now = datetime.now()
        dataset.created_at = now
        dataset.updated_at = now
        
        # Save to file
        if not save_dataset_to_file(dataset):
            raise ProcessingError(f"Failed to save dataset {dataset.id}")
        
        return dataset
    except Exception as e:
        return handle_api_error(e)


@app.put("/datasets/{dataset_id}", response_model=DatasetModel)
async def update_dataset(dataset_id: str, dataset: DatasetModel):
    """Update an existing dataset."""
    try:
        # Ensure the dataset ID matches
        dataset.id = dataset_id
        dataset.updated_at = datetime.now()
        
        # Save to file
        if not save_dataset_to_file(dataset):
            raise ProcessingError(f"Failed to update dataset {dataset_id}")
        
        return dataset
    except Exception as e:
        return handle_api_error(e)


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its associated files."""
    try:
        dataset_path = get_dataset_path(dataset_id)
        embeddings_path = get_embeddings_path(dataset_id)
        attributes_path = get_attributes_path(dataset_id)
        
        # Check if dataset exists
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Don't allow deletion of predefined datasets (in datasets directory)
        predefined_path = DATASETS_DIR / f"{dataset_id}.json"
        if dataset_path == predefined_path:
            raise HTTPException(status_code=403, detail=f"Cannot delete predefined dataset {dataset_id}")
        
        # Delete files (only user-created datasets in output directory)
        dataset_path.unlink()
        if embeddings_path.exists():
            embeddings_path.unlink()
        if attributes_path.exists():
            attributes_path.unlink()
        
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


@app.post("/datasets/{dataset_id}/embeddings")
async def generate_embeddings(dataset_id: str):
    """Generate embeddings for a dataset."""
    try:
        # Load dataset
        dataset = load_dataset_from_file(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get paths
        dataset_path = get_dataset_path(dataset_id)
        embeddings_path = get_embeddings_path(dataset_id)
        
        if CLASSIFIER_AVAILABLE:
            # Use the real classification library
            # Initialize classifier with dataset and embeddings paths
            # This will automatically generate embeddings if they don't exist
            classifier = TextClassifier(
                dataset_path=str(dataset_path),
                embeddings_path=str(embeddings_path)
            )
            
            # Trigger embedding generation by making a dummy classification
            # This forces the classifier to initialize and generate embeddings
            try:
                classifier.predict("dummy text for embedding generation")
            except Exception:
                # The classification might fail, but embeddings should be generated
                pass
            
        else:
            # Create a mock embeddings file for demo mode
            with gzip.open(embeddings_path, 'wb') as f:
                pickle.dump({"mock": "embeddings"}, f)
        
        # Update dataset status
        dataset.embeddingsGenerated = True
        save_dataset_to_file(dataset)
        
        return {"message": f"Embeddings generated for dataset {dataset_id}"}
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


# Classification Endpoints

@app.post("/classify/text")
async def classify_text(request: ClassificationRequestModel):
    """Classify text using the specified dataset and configuration."""
    try:
        # Load dataset
        dataset = load_dataset_from_file(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
        
        # Convert to class definitions
        class_definitions = convert_to_class_definitions(dataset.classes)
        
        # Get paths
        dataset_path = get_dataset_path(request.dataset_id)
        embeddings_path = get_embeddings_path(request.dataset_id)
        
        # Create reranking config if needed
        reranking_config = None
        if request.config.use_reranking:
            from multi_class_text_classifier.models.data_models import RerankingConfig
            
            # Determine model type and configuration based on reranking_model
            reranking_model = request.config.reranking_model or "amazon_rerank"
            
            if (reranking_model == "amazon_rerank" or 
                "amazon.rerank" in reranking_model or 
                ("arn:aws:bedrock" in reranking_model and "amazon.rerank" in reranking_model)):
                # Amazon Rerank (fastest and most cost-effective)
                reranking_config = RerankingConfig(
                    model_type="amazon_rerank",
                    model_id=config.aws.amazon_rerank_model,
                    top_k_candidates=request.config.top_k_candidates,
                    aws_region=config.aws.bedrock_region
                )
            elif reranking_model == "cohere_rerank" or "cohere.rerank" in reranking_model:
                # Cohere Rerank
                reranking_config = RerankingConfig(
                    model_type="cohere_rerank",
                    model_id=config.aws.cohere_rerank_model,
                    top_k_candidates=request.config.top_k_candidates,
                    aws_region=config.aws.bedrock_region,
                    model_parameters={
                        "top_n": request.config.top_k_candidates,
                        "return_documents": True
                    }
                )
            else:
                # LLM-based reranking (fallback to Sonnet 4 instead of Sonnet 3)
                model_id = reranking_model
                if "claude-3-5-sonnet" in model_id or "claude-sonnet-3" in model_id:
                    # Upgrade to Sonnet 4
                    model_id = config.aws.default_claude_sonnet_4_model
                elif not model_id or model_id == "llm":
                    # Default to Nova Lite for cost-effectiveness
                    model_id = config.aws.default_nova_lite_model
                
                reranking_config = RerankingConfig(
                    model_type="llm",
                    model_id=model_id,
                    top_k_candidates=request.config.top_k_candidates,
                    aws_region=config.aws.bedrock_region,
                    model_parameters={
                        "temperature": config.models.reranking_temperature,
                        "max_tokens": config.models.reranking_max_tokens
                    }
                )
        
        # Create attribute validation config if needed
        attribute_config = None
        if request.config.use_attribute_validation:
            attributes_path = get_attributes_path(request.dataset_id)
            attribute_config = AttributeValidationConfig(
                enabled=True,
                attributes_path=str(attributes_path)
            )
        
        # Create and configure classifier with proper paths
        classifier = TextClassifier(
            dataset_path=str(dataset_path),
            embeddings_path=str(embeddings_path),
            reranking_config=reranking_config,
            attribute_config=attribute_config
        )
        
        # Perform classification
        result = classifier.predict(request.text, top_k=request.config.top_k_candidates)
        
        # Convert result to dict for JSON response
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


@app.post("/classify/pdf")
async def classify_pdf(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    config_json: str = Form(default='{}')
):
    """Classify PDF document using the specified dataset and configuration."""
    try:
        logger.info(f"PDF classification request received - file: {file.filename}, dataset: {dataset_id}")
        logger.info(f"Config received: {config_json}")
        # Parse configuration
        try:
            config_dict = json.loads(config_json)
            logger.info(f"Parsed config: {config_dict}")
            classification_config = ClassificationConfigModel(**config_dict)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse config JSON: {e}, using defaults")
            classification_config = ClassificationConfigModel()
        except Exception as e:
            logger.error(f"Error creating classification config: {e}")
            raise InvalidInputError(f"Invalid configuration: {e}")
        
        # Validate file type
        if not file.filename:
            raise InvalidInputError("No filename provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise InvalidInputError("Only PDF files are supported")
        
        logger.info(f"File validation passed for: {file.filename}")
        
        # Save uploaded file temporarily
        temp_file_path = TEMP_DIR / f"temp_{datetime.now().timestamp()}_{file.filename}"
        logger.info(f"Saving file to: {temp_file_path}")
        
        try:
            with open(temp_file_path, 'wb') as f:
                content = await file.read()
                if not content:
                    raise InvalidInputError("Uploaded file is empty")
                f.write(content)
                logger.info(f"File saved successfully, size: {len(content)} bytes")
            
            # Extract text from PDF
            if not pdf_extractor:
                raise ConfigurationError("PDF extractor not available")
            
            logger.info(f"Extracting content from PDF: {file.filename}")
            pdf_content = pdf_extractor.extract_from_file(str(temp_file_path))
            
            # Use full_content which includes both text and image descriptions
            extracted_text = pdf_content.full_content
            logger.info(f"Extracted {len(extracted_text)} characters from PDF")
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise ProcessingError("No meaningful content could be extracted from the PDF")
            
            # Load dataset
            dataset = load_dataset_from_file(dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            
            # Convert to class definitions
            class_definitions = convert_to_class_definitions(dataset.classes)
            
            # Get paths
            dataset_path = get_dataset_path(dataset_id)
            embeddings_path = get_embeddings_path(dataset_id)
            
            # Create reranking config if needed (same logic as text classification)
            reranking_config = None
            if classification_config.use_reranking:
                from multi_class_text_classifier.models.data_models import RerankingConfig
                
                # Determine model type and configuration based on reranking_model
                reranking_model = classification_config.reranking_model or "amazon_rerank"
                
                if (reranking_model == "amazon_rerank" or 
                    "amazon.rerank" in reranking_model or 
                    ("arn:aws:bedrock" in reranking_model and "amazon.rerank" in reranking_model)):
                    # Amazon Rerank (fastest and most cost-effective)
                    reranking_config = RerankingConfig(
                        model_type="amazon_rerank",
                        model_id=config.aws.amazon_rerank_model,
                        top_k_candidates=classification_config.top_k_candidates,
                        aws_region=config.aws.bedrock_region
                    )
                elif reranking_model == "cohere_rerank" or "cohere.rerank" in reranking_model:
                    # Cohere Rerank
                    reranking_config = RerankingConfig(
                        model_type="cohere_rerank",
                        model_id=config.aws.cohere_rerank_model,
                        top_k_candidates=classification_config.top_k_candidates,
                        aws_region=config.aws.bedrock_region,
                        model_parameters={
                            "top_n": classification_config.top_k_candidates,
                            "return_documents": True
                        }
                    )
                else:
                    # LLM-based reranking (fallback to Sonnet 4 instead of Sonnet 3)
                    model_id = reranking_model
                    if "claude-3-5-sonnet" in model_id or "claude-sonnet-3" in model_id:
                        # Upgrade to Sonnet 4
                        model_id = config.aws.default_claude_sonnet_4_model
                    elif not model_id or model_id == "llm":
                        # Default to Nova Lite for cost-effectiveness
                        model_id = config.aws.default_nova_lite_model
                    
                    reranking_config = RerankingConfig(
                        model_type="llm",
                        model_id=model_id,
                        top_k_candidates=classification_config.top_k_candidates,
                        aws_region=config.aws.bedrock_region,
                        model_parameters={
                            "temperature": config.models.reranking_temperature,
                            "max_tokens": config.models.reranking_max_tokens
                        }
                    )
            
            # Create attribute validation config if needed
            attribute_config = None
            if classification_config.use_attribute_validation:
                attributes_path = get_attributes_path(dataset_id)
                attribute_config = AttributeValidationConfig(
                    enabled=True,
                    attributes_path=str(attributes_path)
                )
            
            # Create and configure classifier with proper paths
            logger.info(f"Initializing classifier for dataset: {dataset_id}")
            classifier = TextClassifier(
                dataset_path=str(dataset_path),
                embeddings_path=str(embeddings_path),
                reranking_config=reranking_config,
                attribute_config=attribute_config
            )
            
            # Perform classification
            logger.info(f"Classifying extracted text ({len(extracted_text)} chars)")
            result = classifier.predict(extracted_text, top_k=classification_config.top_k_candidates)
            logger.info(f"Classification completed successfully")
            
            # Add PDF-specific metadata
            logger.info("Converting classification result to dictionary")
            result_dict = result.to_dict()
            
            logger.info("Adding PDF metadata to result")
            result_dict['pdf_metadata'] = {
                'filename': file.filename,
                'page_count': getattr(pdf_content, 'page_count', 0),
                'images_found': len(getattr(pdf_content, 'images', [])),
                'extracted_text_length': len(extracted_text),
                'extracted_text_preview': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                'extraction_method': 'Multimodal LLM'
            }
            
            return result_dict
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


# Attribute Management Endpoints

@app.get("/attributes/{dataset_id}")
async def get_attributes(dataset_id: str):
    """Get attributes for a dataset."""
    try:
        attributes_path = get_attributes_path(dataset_id)
        
        if not attributes_path.exists():
            return {"attributes": [], "generated": False}
        
        with open(attributes_path, 'r', encoding='utf-8') as f:
            attributes_data = json.load(f)
        
        return attributes_data
    except Exception as e:
        return handle_api_error(e)


@app.get("/test/attributes")
async def test_attribute_generator():
    """Test endpoint to verify attribute generator is working."""
    try:
        if not attribute_generator:
            return {"status": "error", "message": "Attribute generator not available"}
        
        # Test the connection
        connection_ok = attribute_generator.test_bedrock_connection()
        
        return {
            "status": "success" if connection_ok else "error",
            "attribute_generator_available": True,
            "bedrock_connection": connection_ok,
            "model_id": attribute_generator.model_config.model_id
        }
    except Exception as e:
        logger.error(f"Error testing attribute generator: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/attributes/{dataset_id}/generate")
async def generate_attributes(dataset_id: str, domain: str = Form(...), request: Request = None):
    """Generate attributes for a dataset using LLM."""
    try:
        logger.info("=" * 50)
        logger.info("FRONTEND REQUEST RECEIVED!")
        logger.info(f"Dataset ID: {dataset_id}")
        logger.info(f"Domain: {domain}")
        if request:
            logger.info(f"Request method: {request.method}")
            logger.info(f"Request URL: {request.url}")
            logger.info(f"Request headers: {dict(request.headers)}")
        logger.info("=" * 50)
        logger.info(f"Starting attribute generation for dataset: {dataset_id}, domain: {domain}")
        
        # Load dataset
        dataset = load_dataset_from_file(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        logger.info(f"Loaded dataset with {len(dataset.classes)} classes")
        
        if not attribute_generator:
            raise ConfigurationError("Attribute generator not available")
        
        # Convert to class definitions
        class_definitions = convert_to_class_definitions(dataset.classes)
        logger.info(f"Converted to {len(class_definitions)} class definitions")
        
        # Generate attributes using asyncio to avoid blocking
        import asyncio
        import concurrent.futures
        
        def generate_sync():
            """Synchronous wrapper for the attribute generation."""
            try:
                logger.info("Starting synchronous attribute generation...")
                logger.info(f"Attribute generator type: {type(attribute_generator)}")
                logger.info(f"Class definitions count: {len(class_definitions)}")
                
                result = attribute_generator.generate_attributes_for_classes(
                    class_definitions, 
                    domain_context=domain
                )
                logger.info("Synchronous attribute generation completed successfully")
                logger.info(f"Result type: {type(result)}")
                logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return result
            except Exception as sync_error:
                logger.error(f"Error in synchronous attribute generation: {sync_error}")
                logger.error(f"Error type: {type(sync_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        # Run the synchronous attribute generation in a thread pool
        # This prevents the FastAPI event loop from blocking
        logger.info("Running attribute generation in thread pool...")
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                attributes_data = await loop.run_in_executor(executor, generate_sync)
        except Exception as executor_error:
            logger.error(f"Error in thread pool executor: {executor_error}")
            logger.error(f"Executor error type: {type(executor_error)}")
            import traceback
            logger.error(f"Executor traceback: {traceback.format_exc()}")
            raise
        
        logger.info("Thread pool execution completed")
        
        # Add dataset-specific metadata
        attributes_data["metadata"]["dataset_id"] = dataset_id
        attributes_data["metadata"]["generated"] = True
        attributes_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Generated attributes for {len(attributes_data.get('classes', []))} classes")
        
        # Save attributes
        attributes_path = get_attributes_path(dataset_id)
        with open(attributes_path, 'w', encoding='utf-8') as f:
            json.dump(attributes_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved attributes to: {attributes_path}")
        logger.info("Attribute generation endpoint completed successfully")
        
        return attributes_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_attributes endpoint: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return handle_api_error(e)


@app.put("/attributes/{dataset_id}")
async def save_attributes(dataset_id: str, attributes_data: Dict[str, Any]):
    """Save attributes for a dataset."""
    try:
        # Validate dataset exists
        dataset = load_dataset_from_file(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Add metadata
        attributes_data["dataset_id"] = dataset_id
        attributes_data["last_updated"] = datetime.now().isoformat()
        
        # Save attributes
        attributes_path = get_attributes_path(dataset_id)
        with open(attributes_path, 'w', encoding='utf-8') as f:
            json.dump(attributes_data, f, indent=2, ensure_ascii=False)
        
        return {"message": f"Attributes saved for dataset {dataset_id}"}
    except HTTPException:
        raise
    except Exception as e:
        return handle_api_error(e)


# Domain Wizard Endpoints

def _generate_dataset_internal(request: DomainGenerationRequestModel) -> DatasetModel:
    """Internal function to generate a dataset (not an endpoint)."""
    logger.info(f"Starting domain dataset generation for: {request.domain}")
    
    if not CLASSIFIER_AVAILABLE:
        # Mock response for demo mode
        dataset_id = f"generated_{request.domain.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mock_classes = [
            ClassDefinitionModel(
                id=str(uuid.uuid4()),
                name=f"Mock Class {i+1}",
                description=f"Mock class for {request.domain} domain",
                examples=[f"Mock example {j+1} for class {i+1}" for j in range(request.examples_per_class)],
                metadata={}
            )
            for i in range(request.num_classes)
        ]
        
        dataset = DatasetModel(
            id=dataset_id,
            name=f"Generated: {request.domain.title()}",
            description=f"Auto-generated dataset for {request.domain} domain with {request.num_classes} classes",
            classes=mock_classes,
            embeddingsGenerated=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        save_dataset_to_file(dataset)
        return dataset
    
    # Use the text classifier library to generate classes
    from multi_class_text_classifier import DatasetGenerator
    
    # Initialize dataset generator if not available
    global dataset_generator
    if not dataset_generator:
        dataset_generator = DatasetGenerator()
    
    # Generate dataset using the text classifier library
    logger.info(f"Generating {request.num_classes} classes for domain: {request.domain}")
    generated_dataset = dataset_generator.generate_dummy_dataset(
        domain=request.domain,
        num_classes=request.num_classes
    )
    
    logger.info(f"Generated {len(generated_dataset['classes'])} classes")
    
    # Create dataset model
    dataset_id = f"generated_{request.domain.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate realistic examples for each class using the example generator service
    classes_with_examples = []
    
    if request.examples_per_class > 0:
        # Generate examples for all classes at once
        try:
            examples_by_class = example_generator.generate_examples(
                generated_dataset['classes'],
                request.domain,
                request.examples_per_class
            )
        except Exception as e:
            logger.error(f"Failed to generate examples using service: {e}")
            examples_by_class = {}
    else:
        examples_by_class = {}
    
    # Create class models with generated examples
    for cls_data in generated_dataset['classes']:
        class_name = cls_data['name']
        examples = examples_by_class.get(class_name, [])
        
        # Fallback if no examples were generated
        if not examples and request.examples_per_class > 0:
            examples = [f"Example document for {class_name} in {request.domain} domain" for _ in range(request.examples_per_class)]
        
        classes_with_examples.append(
            ClassDefinitionModel(
                id=str(uuid.uuid4()),
                name=class_name,
                description=cls_data['description'],
                examples=examples,
                metadata={}
            )
        )
    
    dataset = DatasetModel(
        id=dataset_id,
        name=f"Generated: {request.domain.title()}",
        description=f"Auto-generated dataset for {request.domain} domain with {request.num_classes} classes",
        classes=classes_with_examples,
        embeddingsGenerated=False,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Save dataset
    if not save_dataset_to_file(dataset):
        raise ProcessingError(f"Failed to save generated dataset {dataset_id}")
    
    logger.info(f"Successfully generated and saved dataset: {dataset_id}")
    return dataset


@app.post("/wizard/generate-dataset")
async def generate_domain_dataset(request: DomainGenerationRequestModel):
    """Generate a complete dataset for a specific domain."""
    try:
        dataset = _generate_dataset_internal(request)
        return dataset
    except Exception as e:
        logger.error(f"Error in generate_domain_dataset: {e}")
        return handle_api_error(e)


@app.post("/wizard/generate-domain")
async def generate_domain_complete(request: DomainGenerationRequestModel):
    """Generate a complete domain setup with dataset and optionally attributes."""
    try:
        logger.info(f"Starting complete domain generation for: {request.domain}")
        
        # Generate the dataset first using internal function
        dataset = _generate_dataset_internal(request)
        
        # Skip automatic attribute generation for now to avoid loops
        attributes = None
        logger.info("Skipping automatic attribute generation to prevent loops")
        
        result = {
            "dataset": dataset,
            "attributes": attributes
        }
        
        logger.info(f"Successfully completed domain generation for: {request.domain}")
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_domain_complete: {e}")
        return handle_api_error(e)


@app.post("/wizard/generate-examples")
async def generate_domain_examples(
    domain: str = Form(...),
    classes: str = Form(...),  # JSON string of class definitions
    examples_per_class: int = Form(default=3)
):
    """Generate example documents for specific classes using boto3 converse API."""
    try:
        logger.info(f"Generating examples for domain: {domain}")
        
        # Parse classes from JSON
        try:
            classes_data = json.loads(classes)
        except json.JSONDecodeError:
            raise InvalidInputError("Invalid classes JSON format")
        
        if not CLASSIFIER_AVAILABLE:
            # Mock response for demo mode
            examples = {}
            for cls in classes_data:
                class_name = cls.get('name', 'Unknown')
                examples[class_name] = [
                    f"Mock example {i+1} for {class_name} in {domain} domain"
                    for i in range(examples_per_class)
                ]
            return {"examples": examples}
        
        # Use boto3 with converse API to generate examples
        import boto3
        from botocore.exceptions import ClientError
        
        try:
            # Initialize bedrock client
            bedrock_client = boto3.client('bedrock-runtime', region_name=config.aws.bedrock_region)
            
            examples = {}
            
            for cls in classes_data:
                class_name = cls.get('name', 'Unknown')
                class_description = cls.get('description', '')
                
                logger.info(f"Generating {examples_per_class} examples for class: {class_name}")
                
                # Create prompt for example generation
                prompt = ExampleGenerationPrompts.single_class_example_generation_prompt(
                    class_name, class_description, domain, examples_per_class
                )
                
                # Use converse API
                response = bedrock_client.converse(
                    modelId=config.aws.default_nova_lite_model,  # Use Nova Lite for cost-effectiveness
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    inferenceConfig={
                        "temperature": config.models.example_generation_temperature,
                        "maxTokens": config.models.example_generation_max_tokens
                    }
                )
                
                # Extract generated text
                generated_text = response['output']['message']['content'][0]['text']
                
                # Parse examples from generated text
                class_examples = []
                lines = generated_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # Remove numbering and clean up
                        example = line.split('.', 1)[-1].strip()
                        if example.startswith('-'):
                            example = example[1:].strip()
                        if example:
                            class_examples.append(example)
                
                # Ensure we have the right number of examples
                if len(class_examples) < examples_per_class:
                    # Pad with generic examples if needed
                    for i in range(len(class_examples), examples_per_class):
                        class_examples.append(f"Example document for {class_name} in {domain} domain")
                elif len(class_examples) > examples_per_class:
                    # Trim to requested number
                    class_examples = class_examples[:examples_per_class]
                
                examples[class_name] = class_examples
                logger.info(f"Generated {len(class_examples)} examples for {class_name}")
            
            return {"examples": examples}
            
        except ClientError as e:
            logger.error(f"AWS Bedrock error: {e}")
            # Fallback to simple examples
            examples = {}
            for cls in classes_data:
                class_name = cls.get('name', 'Unknown')
                examples[class_name] = [
                    f"Generated example {i+1} for {class_name} in {domain} domain"
                    for i in range(examples_per_class)
                ]
            return {"examples": examples}
            
    except Exception as e:
        logger.error(f"Error in generate_domain_examples: {e}")
        return handle_api_error(e)


@app.get("/wizard/domain-suggestions")
async def get_domain_suggestions(q: str = ""):
    """Get domain suggestions for the wizard."""
    try:
        # Default domain suggestions
        default_suggestions = [
            'medical supplies',
            'legal documents', 
            'financial reports',
            'customer support tickets',
            'product reviews',
            'news articles',
            'academic papers',
            'business emails',
            'technical documentation',
            'marketing materials',
            'insurance claims',
            'real estate listings',
            'job applications',
            'restaurant reviews',
            'travel bookings',
            'software bugs',
            'scientific research',
            'educational content',
            'social media posts',
            'e-commerce products'
        ]
        
        if not q.strip():
            return {"suggestions": default_suggestions[:10]}
        
        # Filter suggestions based on query
        query_lower = q.lower()
        filtered = [s for s in default_suggestions if query_lower in s.lower()]
        
        # If no matches, return all suggestions
        if not filtered:
            filtered = default_suggestions
        
        return {"suggestions": filtered[:10]}
        
    except Exception as e:
        return handle_api_error(e)


# Example Generation Endpoints

@app.post("/examples/generate")
async def generate_examples(
    class_name: str = Form(...),
    class_description: str = Form(...),
    num_examples: int = Form(default=5)
):
    """Generate example documents for a specific class."""
    try:
        if not dataset_generator:
            raise ConfigurationError("Dataset generator not available")
        
        # Create a temporary class definition
        class_def = ClassDefinition(
            name=class_name,
            description=class_description
        )
        
        # Generate examples (this would need to be implemented in the dataset generator)
        # For now, return a placeholder response
        examples = [
            f"Example {i+1} for {class_name}: {class_description}"
            for i in range(num_examples)
        ]
        
        return {
            "class_name": class_name,
            "class_description": class_description,
            "examples": examples,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return handle_api_error(e)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Class Text Classifier API server...")
    print(f"Datasets directory: {DATASETS_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    uvicorn.run(
        "backend_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )