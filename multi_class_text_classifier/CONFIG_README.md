# Multi-Class Text Classifier Configuration

This document describes the configuration system for the multi-class text classifier library.

## Configuration Files

### `config.py`
Contains the configuration classes and settings for the classifier library:
- **AWSConfig**: AWS regions and model IDs
- **RetryConfig**: AWS service retry behavior
- **PDFConfig**: PDF extraction parameters
- **DatasetConfig**: Dataset generation settings
- **SimilarityConfig**: Similarity search defaults
- **ValidationConfig**: Validation and fallback scores
- **ClassifierConfig**: Main configuration container

### `prompts.py`
Contains all LLM prompts and tool definitions:
- **AttributeGenerationPrompts**: Prompts for attribute generation
- **AttributeEvaluationPrompts**: Prompts for attribute evaluation
- **RerankingPrompts**: Prompts for LLM-based reranking
- **DatasetGenerationPrompts**: Prompts for dataset generation
- **ToolDefinitions**: Tool definitions for structured LLM responses

## Environment Variables

You can override configuration values using environment variables:

### AWS Configuration
- `AWS_BEDROCK_REGION`: Bedrock service region (default: us-west-2)

### Model IDs
- `AWS_NOVA_LITE_MODEL`: Nova Lite model ID (default: us.amazon.nova-lite-v1:0)
- `AWS_NOVA_PRO_MODEL`: Nova Pro model ID (default: us.amazon.nova-pro-v1:0)
- `AWS_CLAUDE_SONNET_4_MODEL`: Claude Sonnet 4 model ID (default: us.anthropic.claude-sonnet-4-20250514-v1:0)

### Retry Configuration
- `AWS_RETRY_MAX_ATTEMPTS`: Maximum retry attempts (default: 10)
- `AWS_RETRY_MODE`: Retry mode (default: standard)
- `AWS_READ_TIMEOUT`: Read timeout in seconds (default: 180)
- `AWS_CONNECT_TIMEOUT`: Connect timeout in seconds (default: 15)
- `AWS_MAX_POOL_CONNECTIONS`: Max pool connections (default: 5)

### Reranking Configuration
- `RERANKING_MAX_RETRIES`: Maximum retries for reranking operations (default: 3)
- `RERANKING_BASE_DELAY`: Base delay for reranking retries in seconds (default: 1.0)
- `RERANKING_READ_TIMEOUT`: Read timeout for reranking operations (default: 30)
- `RERANKING_CONNECT_TIMEOUT`: Connect timeout for reranking operations (default: 10)

### Attribute Generation Configuration
- `ATTR_GEN_MAX_RETRIES`: Maximum retries for attribute generation (default: 3)
- `ATTR_GEN_RETRY_DELAY`: Base delay for attribute generation retries (default: 5)
- `ATTR_GEN_CLASS_DELAY`: Delay between processing classes (default: 3)

### PDF Extraction Configuration
- `PDF_MODEL_ID`: Model ID for PDF extraction (default: us.amazon.nova-lite-v1:0)
- `PDF_AWS_REGION`: AWS region for PDF processing (default: us-west-2, now consolidated with AWS_BEDROCK_REGION)
- `PDF_MAX_TOKENS`: Maximum tokens for PDF model inference (default: 4000)
- `PDF_TEMPERATURE`: Temperature for PDF model inference (default: 0.1)
- `PDF_TOP_P`: Top-p for PDF model inference (default: 0.9)
- `PDF_MAX_IMAGE_SIZE`: Maximum image dimension for processing (default: 1024)
- `PDF_IMAGE_QUALITY`: JPEG quality for image compression (default: 85)
- `PDF_MIN_IMAGE_SIZE`: Minimum image size to process (default: 50)
- `PDF_MAX_IMAGES_PER_PAGE`: Maximum images to process per page (default: 10)
- `PDF_MAX_TOTAL_IMAGES`: Maximum total images to process (default: 50)
- `PDF_READ_TIMEOUT`: Read timeout for PDF operations (default: 60)
- `PDF_CONNECT_TIMEOUT`: Connect timeout for PDF operations (default: 10)
- `PDF_DEFAULT_CONFIDENCE`: Default confidence score for OCR (default: 0.8)

### Dataset Generation Configuration
- `DATASET_MODEL_ID`: Model ID for dataset generation (default: us.amazon.nova-lite-v1:0)
- `DATASET_MAX_TOKENS`: Maximum tokens for dataset generation (default: 4000)
- `DATASET_DEFAULT_NUM_CLASSES`: Default number of classes to generate (default: 50)
- `DATASET_JSON_INDENT`: JSON indentation for saved files (default: 2)

### Similarity Search Configuration
- `SIMILARITY_DEFAULT_TOP_K`: Default number of top candidates (default: 5)
- `SIMILARITY_ZERO_NORM_EPSILON`: Epsilon for numerical stability (default: 1e-8)

### Validation Configuration
- `VALIDATION_DEFAULT_PASS_SCORE`: Default score for passing validation (default: 1.0)
- `VALIDATION_DEFAULT_FAIL_SCORE`: Default score for failing validation (default: 0.0)
- `VALIDATION_FALLBACK_RERANK_SCORE`: Fallback score for reranking failures (default: 0.1)

## Usage

### Basic Configuration Access
```python
from multi_class_text_classifier.config import config

# Access AWS settings
region = config.aws.bedrock_region
model_id = config.aws.default_claude_sonnet_4_model

# Access retry settings
max_attempts = config.retry.max_attempts
```

### Using Prompts
```python
from multi_class_text_classifier.prompts import AttributeGenerationPrompts

prompt = AttributeGenerationPrompts.single_class_generation_prompt(
    class_name="Invoice",
    class_description="Financial billing document",
    domain_context="Financial documents"
)
```

### Using Tool Definitions
```python
from multi_class_text_classifier.prompts import ToolDefinitions

tool_def = ToolDefinitions.generate_class_attributes_tool()
```

## Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Modify values in `.env` as needed for your environment

3. The configuration will automatically load environment variables on startup

## Independence from UI Backend

This configuration is completely independent of the UI backend configuration. The classifier library can be used standalone without any dependencies on the UI components.

## Migration Notes

Previously, the attribute generator tried to import configuration from the UI backend. This has been changed to use its own independent configuration system, ensuring the library remains modular and reusable.