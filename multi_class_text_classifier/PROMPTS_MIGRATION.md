# Prompts Migration Summary

This document summarizes the migration of hardcoded prompts from the multi-class text classifier library to a centralized prompts file.

## Files Created

### `multi_class_text_classifier/prompts.py`
Centralized prompts file containing all LLM prompts and tool definitions used throughout the library.

## Hardcoded Prompts Migrated

### 1. **Attribute Generation Prompts**

**Before:** Hardcoded in `config.py`
```python
return f"""You are an expert in document classification. Generate logical attribute requirements for this document class.

Class: {class_name}
Description: {class_description}{domain_info}

INSTRUCTIONS:
1. Create specific, measurable attributes...
```

**After:** Centralized in `prompts.py`
```python
AttributeGenerationPrompts.single_class_generation_prompt(
    class_name, class_description, domain_context
)
```

### 2. **Attribute Evaluation Prompts**

**Before:** Hardcoded in `attribute_evaluator.py`
```python
system_prompt="You are an expert document classifier. Evaluate whether text satisfies specific conditions."

prompt = f"""You are evaluating whether a document satisfies a specific condition for the class "{class_name}".

TEXT TO EVALUATE:
{text}

CONDITION TO CHECK:
{condition}
```

**After:** Centralized in `prompts.py`
```python
AttributeEvaluationPrompts.system_prompt()
AttributeEvaluationPrompts.single_condition_evaluation_prompt(text, class_name, condition)
```

### 3. **Reranking Prompts**

**Before:** Hardcoded in `llm_reranker.py`
```python
system_prompt="""You are a text classification expert. Your task is to rerank classification candidates based on how well they match the input text.

You will be given:
1. An input text to classify
2. A list of candidate classes with their descriptions
```

**After:** Centralized in `prompts.py`
```python
RerankingPrompts.system_prompt()
RerankingPrompts.reranking_prompt(input_text, candidate_dicts)
```

### 4. **Dataset Generation Prompts**

**Before:** Hardcoded in `dataset_generator.py`
```python
system_prompt="You are a helpful assistant that generates diverse classification datasets for given domains. Always respond with valid JSON format."

prompt = f"""Generate {num_classes} diverse and distinct classification classes for the domain: {domain}

Requirements:
1. Each class should have a clear, specific name (2-4 words maximum)
2. Each class should have a detailed description (20-50 words)...
```

**After:** Centralized in `prompts.py`
```python
DatasetGenerationPrompts.system_prompt()
DatasetGenerationPrompts.dataset_generation_prompt(domain, num_classes)
```

### 5. **Tool Definitions**

**Before:** Hardcoded in `config.py`
```python
return {
    "toolSpec": {
        "name": "generate_class_attributes",
        "description": "Generate attribute definition for a document class",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "required_attributes": {...}
                }
            }
        }
    }
}
```

**After:** Centralized in `prompts.py`
```python
ToolDefinitions.generate_class_attributes_tool()
ToolDefinitions.rerank_candidates_tool()
ToolDefinitions.validate_attributes_tool()
```

## Files Modified

### Core Library Files
- `multi_class_text_classifier/prompts.py` - **NEW**: Centralized prompts file
- `multi_class_text_classifier/config.py` - Removed prompts and tool definitions
- `multi_class_text_classifier/attribute_generator.py` - Updated imports
- `multi_class_text_classifier/attribute_evaluator.py` - Updated to use prompts, removed hardcoded method
- `multi_class_text_classifier/llm_reranker.py` - Updated to use prompts
- `multi_class_text_classifier/dataset_generator.py` - Updated to use prompts, removed hardcoded method

### Documentation
- `multi_class_text_classifier/CONFIG_README.md` - Updated to reference prompts file
- `multi_class_text_classifier/PROMPTS_MIGRATION.md` - **NEW**: This summary document

## Prompts Organization

The prompts are organized into logical classes:

### `AttributeGenerationPrompts`
- `single_class_generation_prompt()` - For generating class attributes

### `AttributeEvaluationPrompts`
- `system_prompt()` - System prompt for evaluation agent
- `single_condition_evaluation_prompt()` - For evaluating single conditions

### `RerankingPrompts`
- `system_prompt()` - System prompt for reranking agent
- `reranking_prompt()` - For LLM-based candidate reranking

### `DatasetGenerationPrompts`
- `system_prompt()` - System prompt for dataset generation agent
- `dataset_generation_prompt()` - For generating classification datasets

### `ToolDefinitions`
- `generate_class_attributes_tool()` - Tool for attribute generation
- `rerank_candidates_tool()` - Tool for candidate reranking
- `validate_attributes_tool()` - Tool for attribute validation

## Benefits Achieved

### 1. **Separation of Concerns**
- Configuration focused on system settings and parameters
- Prompts focused on LLM interactions and templates
- Clear separation between technical configuration and prompt engineering

### 2. **Centralized Prompt Management**
- Single source of truth for all prompts
- Easy to modify prompts without touching business logic
- Version control for prompt changes
- Consistent prompt formatting and structure

### 3. **Maintainability**
- Easier to update prompts for better performance
- A/B testing of different prompt variations
- Clear organization by functionality
- Reduced code duplication

### 4. **Prompt Engineering Focus**
- Prompts can be optimized independently of code
- Easy to experiment with different prompt strategies
- Clear documentation of prompt purposes and parameters
- Consistent tool definitions across the library

### 5. **Independence**
- Library remains completely independent of UI backend
- No cross-dependencies between prompt systems
- Can be used standalone in other projects

## Usage Examples

### Basic Prompt Usage
```python
from multi_class_text_classifier.prompts import AttributeGenerationPrompts

# Generate attribute generation prompt
prompt = AttributeGenerationPrompts.single_class_generation_prompt(
    class_name="Invoice",
    class_description="Financial billing document",
    domain_context="Financial documents"
)
```

### System Prompt Usage
```python
from multi_class_text_classifier.prompts import RerankingPrompts

# Get system prompt for reranking agent
system_prompt = RerankingPrompts.system_prompt()
```

### Tool Definition Usage
```python
from multi_class_text_classifier.prompts import ToolDefinitions

# Get tool definition for structured responses
tool_def = ToolDefinitions.generate_class_attributes_tool()
```

### Reranking Prompt Usage
```python
from multi_class_text_classifier.prompts import RerankingPrompts

# Generate reranking prompt
candidates = [
    {'name': 'Invoice', 'description': 'Financial billing document'},
    {'name': 'Receipt', 'description': 'Proof of purchase document'}
]
prompt = RerankingPrompts.reranking_prompt("Payment document", candidates)
```

## Migration Impact

- **No Breaking Changes**: All functionality remains the same
- **Improved Organization**: Clear separation between config and prompts
- **Better Maintainability**: Easier to modify and optimize prompts
- **Enhanced Flexibility**: Prompts can be updated independently
- **Professional Structure**: Industry-standard separation of concerns

## Future Enhancements

With prompts now centralized, future enhancements become easier:

1. **Prompt Versioning**: Different prompt versions for different use cases
2. **A/B Testing**: Easy to test different prompt variations
3. **Localization**: Support for prompts in different languages
4. **Dynamic Prompts**: Runtime prompt selection based on context
5. **Prompt Optimization**: Systematic optimization of prompt performance