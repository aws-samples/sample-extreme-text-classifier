# Getting Started Examples

This directory contains simple, functional examples to help you get started with the Multi-Class Text Classifier.

## Examples

1. **`1_basic_classification.py`** - Basic document classification using the document classification dataset
2. **`2_classification_with_reranking.py`** - Same classification but with LLM reranking for improved accuracy
3. **`3_pdf_document_classification.py`** - Extract content from PDF files and classify documents
4. **Attribute Validation Workflow** - Enhanced classification with attribute validation:
   - **`4a_simple_attribute_validation.py`** - Simple attribute validation with a single test case
   - **`4b_comprehensive_attribute_validation.py`** - Comprehensive attribute validation across multiple document types
   - **`4c_attribute_generation_example.py`** - Generate attributes automatically using Claude Sonnet 4
5. **Medical Supplies Workflow** - Complete workflow showing dataset generation and usage:
   - **`5a_generate_medical_dataset.py`** - Generate a medical supplies dataset
   - **`5b_classify_medical_supplies.py`** - Use the dataset for classification (generates embeddings)
   - **`5c_classify_with_existing_embeddings.py`** - Reuse existing embeddings for faster initialization

## Quick Start

1. **Start here**: `1_basic_classification.py` - See basic document classification in action
2. **Add reranking**: `2_classification_with_reranking.py` - See how LLM reranking improves results
3. **PDF processing**: `3_pdf_document_classification.py` - Extract and classify PDF content (requires AWS credentials)
4. **Attribute validation**: Run the attribute validation examples (4a → 4b → 4c) to see enhanced classification
5. **Complete workflow**: Run the medical supplies examples (5a → 5b → 5c) to see the full dataset generation process

## File Structure

- **Input files**: Uses datasets from the `datasets/` folder
- **Generated files**: Medical supplies dataset and embeddings are saved to `datasets/` folder
- **Document dataset**: `datasets/document_classification_dataset.json` with 6 document types

## Key Learning Points

- **Example 1**: Basic classification workflow
- **Example 2**: How reranking improves accuracy with LLM semantic understanding
- **Example 3**: PDF content extraction and classification
- **Example 4a**: Simple attribute validation for enhanced confidence
- **Example 4b**: Comprehensive attribute validation across multiple document types
- **Example 4c**: Automatic attribute generation
- **Example 5a**: How to generate domain-specific datasets
- **Example 5b**: First-time classification generates embeddings
- **Example 5c**: Reusing embeddings for faster subsequent runs

## Requirements

- Python dependencies: Install with `pip install -r requirements.txt`
- AWS credentials: Required for invoking foundation models on Bedrock
- The examples use existing files in `datasets/` folder for immediate functionality

## Dataset Information

- **Document Classification**: 6 classes (Invoice, Technical Manual, Legal Contract, Business Report, Research Paper, Marketing Material)
- **Medical Supplies**: Generated dynamically with 20 classes of medical equipment and supplies