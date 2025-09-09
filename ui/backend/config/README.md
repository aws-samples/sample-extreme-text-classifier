# Configuration Guide

This directory contains configuration files for the multi-class text classifier backend.

## Files

### `system_config.py`
Contains all system-level configuration including:
- **AWS Configuration**: Regions, model IDs, service endpoints
- **Model Configuration**: Temperature, max tokens, and other model parameters
- **API Configuration**: CORS origins, file size limits, cleanup settings
- **Retry Configuration**: AWS service retry behavior and timeouts

### `prompts.py`
Contains all LLM prompts and tool definitions:
- **Attribute Generation Prompts**: For generating class attributes
- **Example Generation Prompts**: For creating document examples
- **Classification Prompts**: For reranking and validation
- **Tool Definitions**: Structured response schemas for LLMs

## Environment Variables

You can override any configuration value using environment variables. See `.env.example` for all available options.

### Key Environment Variables

#### AWS Configuration
- `AWS_BEDROCK_REGION`: Bedrock service region (default: us-west-2)

#### Model IDs
- `AWS_NOVA_LITE_MODEL`: Nova Lite model ID
- `AWS_NOVA_PRO_MODEL`: Nova Pro model ID  
- `AWS_CLAUDE_SONNET_4_MODEL`: Claude Sonnet 4 model ID
- `AWS_AMAZON_RERANK_MODEL`: Amazon Rerank model ARN
- `AWS_COHERE_RERANK_MODEL`: Cohere Rerank model ID

#### Model Parameters
- `MODEL_DEFAULT_TEMPERATURE`: Default temperature for all models
- `MODEL_ATTR_GEN_TEMPERATURE`: Temperature for attribute generation
- `MODEL_EXAMPLE_GEN_TEMPERATURE`: Temperature for example generation
- `MODEL_RERANKING_TEMPERATURE`: Temperature for reranking

## Usage

### In Python Code

```python
from config import config

# Access AWS configuration
region = config.aws.bedrock_region
model_id = config.aws.default_nova_lite_model

# Access model parameters
temperature = config.models.default_temperature
max_tokens = config.models.default_max_tokens

# Access prompts
from config import AttributeGenerationPrompts
prompt = AttributeGenerationPrompts.single_class_generation_prompt(
    class_name="Invoice",
    class_description="Financial document for billing",
    domain_context="Financial documents"
)
```

### Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Modify values in `.env` as needed for your environment

3. The configuration will automatically load environment variables on startup

## Benefits

### Centralized Configuration
- All hardcoded values moved to configuration files
- Easy to modify without changing code
- Environment-specific overrides via environment variables

### Maintainability
- Single source of truth for all settings
- Type-safe configuration with dataclasses
- Clear separation between prompts and system config

### Flexibility
- Easy to switch between different models
- Region-specific deployments
- Development vs production configurations

## Migration from Hardcoded Values

The following hardcoded values have been moved to configuration:

### AWS Regions
- `us-west-2` → `config.aws.bedrock_region`

### Model IDs
- `us.amazon.nova-lite-v1:0` → `config.aws.default_nova_lite_model`
- `us.amazon.nova-pro-v1:0` → `config.aws.default_nova_pro_model`
- `us.anthropic.claude-sonnet-4-20250514-v1:0` → `config.aws.default_claude_sonnet_4_model`

### Model Parameters
- Temperature values → `config.models.*_temperature`
- Max tokens → `config.models.*_max_tokens`

### Prompts
- All LLM prompts moved to `prompts.py` with parameterized templates
- Tool definitions centralized for reuse

## Adding New Configuration

To add new configuration options:

1. Add the field to the appropriate dataclass in `system_config.py`
2. Add environment variable support in the `from_env()` method
3. Document the new option in `.env.example`
4. Update this README with usage examples