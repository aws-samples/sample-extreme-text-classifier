# Extreme Text Classifier

A Python text classifier for large-scale multi-class classification tasks with 1000+ classes using Amazon Bedrock. Supports basic classification, reranking with LLMs, and PDF document processing.

**For a live demo of the library's capabilities, see [ui/README.md](ui/README.md) for the interactive frontend showcase.**

## Features

- **Large-Scale Classification**: Handle 1000+ classes with high accuracy
- **LLM-Based Reranking**: Improve accuracy using Amazon Bedrock models
- **Attribute Validation**: Enhanced confidence scoring with detailed condition checking
- **PDF Document Processing**: Extract and classify content from PDF files using multimodal LLMs
- **Dataset Generation**: Create synthetic datasets for any domain
- **Step-by-Step Scoring**: Transparent similarity, rerank, and attribute scores

## How It Works

The classifier uses a **3-step pipeline** that combines semantic understanding with optional quality enhancements:

```
                   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
  Input Text   ───▶│ Semantic Search │───▶│  LLM Reranking  │───▶│ Attribute Validation│
  "Invoice..."     │                 │    │   (Optional)    │    │    (Optional)       │
                   │ Find Similar    │    │ Refine Top-K    │    │ Validate Rules      │
                   │ Classes 1000+   │    │ Classes 1-5     │    │ Business Logic      │
                   └─────────────────┘    └─────────────────┘    └─────────────────────┘
                            ↓                      ↓                       ↓                       
                    Similarity Score          Rerank Score           Attribute Score        
```

### 1. 🔍 Semantic Search (Required)
Uses embedding-based similarity to find the most relevant classes from your dataset. This step handles the core classification by computing cosine similarity between the input text and all class descriptions. Fast and scalable across 1000+ classes.

### 2. 🎯 LLM Reranking (Optional)
Refines the top candidates using Amazon Bedrock models for deeper semantic understanding. This step improves accuracy by having LLMs evaluate the contextual relevance of each candidate, focusing computational resources on the most promising options.

### 3. ✅ Attribute Validation (Optional)
Validates that the predicted class matches specific business rules and conditions. This step provides additional confidence by checking if the text satisfies class-specific requirements using LLM-based condition evaluation and logical operators.

#### Example in practice
If a document is classified as an Invoice, it must have:
  * A monetary amount to be paid (e.g., "$500.00", "Total: $1,250")
  * Both parties identified (e.g., "From: ABC Corp" and "To: XYZ Inc")
  * Either a request for payment OR proof of completed payment

This validation catches instances where receipts have been incorrectly classified as invoices.

### Why This Pipeline Works

- **⚡ Scalable**: Embedding search handles 1000+ classes in milliseconds, with optional LLM steps only processing top candidates
- **🎯 Accurate**: Multi-step validation catches edge cases that pure similarity might miss
- **🔒 Reliable**: Attribute validation ensures predictions meet business requirements, not just semantic similarity
- **📊 Transparent**: Each step provides its own confidence score, allowing you to understand and tune the decision process
- **🛠️ Flexible**: Use just semantic search for speed, or add reranking and validation for maximum accuracy

## Installation

1. **Prerequisites**:
   * Python 3.8+
   * AWS CLI installed and configured
   * AWS account with Bedrock access and required models activated (Amazon Nova Lite, Amazon Nova Pro, Amazon Rerank, Cohere Rerank, Anthropic Claude Sonnet 4)

2. **Install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up AWS credentials**:
   ```bash
   aws configure
   ```

4. **Run examples**:
   ```bash
   # Try the getting started examples
   python examples/getting_started/1_basic_classification.py
   python examples/getting_started/2_classification_with_reranking.py
   python examples/getting_started/3_pdf_document_classification.py
   ```

## Quick Start

### Basic Classification Examples

#### 1. Basic Text Classification

Start with simple text classification using a pre-built dataset. The classifier uses embeddings to find the most similar class for your input text.

*See full example: [examples/getting_started/1_basic_classification.py](examples/getting_started/1_basic_classification.py)*

```python
from multi_class_text_classifier import TextClassifier

classifier = TextClassifier(
    dataset_path="datasets/document_classification_dataset.json",
    embeddings_path="datasets/document_classification_embeddings.pkl.gz"
)

result = classifier.predict("Please remit payment of $1,250.00 within 30 days.")
print(f"Predicted Class: {result.predicted_class.name}")
print(f"Confidence: {result.effective_score:.4f}")
```

#### 2. Classification with Reranking

Improve classification accuracy by adding a reranking step using Amazon Bedrock models. This refines the initial similarity search results.

*See full example: [examples/getting_started/2_classification_with_reranking.py](examples/getting_started/2_classification_with_reranking.py)*

```python
from multi_class_text_classifier import TextClassifier
from multi_class_text_classifier.models.data_models import RerankingConfig

reranking_config = RerankingConfig(
    model_type="amazon_rerank",
    model_id="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    top_k_candidates=5,
    aws_region="us-west-2"
)

classifier = TextClassifier(
    embeddings_path="datasets/document_classification_embeddings.pkl.gz",
    reranking_config=reranking_config
)

result = classifier.predict("This quarterly financial report shows revenue growth.")
print(f"Predicted Class: {result.predicted_class.name}")
print(f"Reranked: {result.reranked}")
```

#### 3. PDF Document Classification

Extract content from PDF files using multimodal LLMs that can process both text and images, then classify the extracted content.

*See full example: [examples/getting_started/3_pdf_document_classification.py](examples/getting_started/3_pdf_document_classification.py)*

```python
from multi_class_text_classifier import create_pdf_extractor, TextClassifier

pdf_extractor = create_pdf_extractor(
    model_id="us.amazon.nova-lite-v1:0"
)

pdf_content = pdf_extractor.extract_from_file("datasets/dummy_invoice.pdf")

classifier = TextClassifier(
    embeddings_path="datasets/document_classification_embeddings.pkl.gz"
)

result = classifier.predict(pdf_content.full_content)
print(f"Document type: {result.predicted_class.name}")
print(f"Confidence: {result.effective_score:.2%}")
```

### Attribute Validation Examples

#### 4a. Simple Attribute Validation

Enhance classification confidence by validating that the predicted class matches specific attributes. This provides additional scoring and detailed feedback on why a classification succeeded or failed.

*See full example: [examples/getting_started/4a_simple_attribute_validation.py](examples/getting_started/4a_simple_attribute_validation.py)*

```python
from multi_class_text_classifier import TextClassifier
from multi_class_text_classifier.models.data_models import (
    AttributeValidationConfig, AttributeEvaluationConfig
)

attribute_config = AttributeValidationConfig(
    enabled=True,
    attributes_path="datasets/document_classification_attributes.json",
    evaluation_config=AttributeEvaluationConfig(
        model_id="us.amazon.nova-lite-v1:0"
    )
)

classifier = TextClassifier(
    dataset_path="datasets/document_classification_dataset.json",
    embeddings_path="datasets/document_classification_embeddings.pkl.gz",
    attribute_config=attribute_config
)

result = classifier.predict("INVOICE #001 - Amount Due: $500 - From: ABC Corp")
print(f"Predicted: {result.predicted_class.name}")
print(f"Similarity Score: {result.similarity_score:.3f}")
print(f"Attribute Score: {result.attribute_score:.3f}")

if result.attribute_validation:
    print(f"Conditions Met: {result.attribute_validation.conditions_met}")
    print(f"Conditions Not Met: {result.attribute_validation.conditions_not_met}")
```

#### 4c. Generate Attributes with LLM

Automatically generate attribute definitions for existing classes using LLMs. This can save you time if you don't have attributes for your classes and want to use attribute validation to enhance classification confidence.

*See full example: [examples/getting_started/4c_attribute_generation_example.py](examples/getting_started/4c_attribute_generation_example.py)*

```python
from multi_class_text_classifier.attribute_generator import AttributeGenerator
from multi_class_text_classifier.dataset_loader import ClassesDataset
from multi_class_text_classifier.models.data_models import AttributeGenerationConfig

# Configure attribute generation
generation_config = AttributeGenerationConfig(
    model_id="us.amazon.nova-pro-v1:0",
    temperature=0.1,
    max_tokens=4000
)

generator = AttributeGenerator(generation_config)

# Load existing classes from dataset
dataset_loader = ClassesDataset("datasets/document_classification_dataset.json")
classes = dataset_loader.load_classes_from_json()

# Generate attributes for all classes (returns standard format)
all_attributes = generator.generate_attributes_for_classes(
    classes,
    domain_context="document classification"
)

# Save generated attributes
generator.save_attributes_to_file(all_attributes, "output/generated_attributes.json")
```

#### 4d. Use Generated Attributes

Use the LLM-generated attributes for classification and validation. This demonstrates the complete workflow from attribute generation to practical usage.

*See full example: [examples/getting_started/4d_use_generated_attributes.py](examples/getting_started/4d_use_generated_attributes.py)*

```python
from multi_class_text_classifier import TextClassifier
from multi_class_text_classifier.models.data_models import (
    AttributeValidationConfig, AttributeEvaluationConfig
)

# Convert generated attributes to standard format
standard_attributes_path = convert_generated_to_standard_format()

attribute_config = AttributeValidationConfig(
    enabled=True,
    attributes_path=standard_attributes_path,
    evaluation_config=AttributeEvaluationConfig(
        model_id="us.amazon.nova-lite-v1:0"
    )
)

classifier = TextClassifier(
    dataset_path="datasets/document_classification_dataset.json",
    embeddings_path="datasets/document_classification_embeddings.pkl.gz",
    attribute_config=attribute_config
)

result = classifier.predict("INVOICE #INV-2024-001 - Amount Due: $2,500.00")
print(f"Predicted: {result.predicted_class.name}")
print(f"Attribute Score: {result.attribute_validation.overall_score:.3f}")
```

## Reranking Models

You can use different types of models for reranking through Amazon Bedrock:

### Dedicated Reranking Models
Models specifically designed for reranking tasks:

```python
# Amazon Rerank
RerankingConfig(
    model_type="amazon_rerank",
    model_id="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
)

# Cohere Rerank
RerankingConfig(
    model_type="cohere_rerank",
    model_id="cohere.rerank-v3-5:0"
)
```

### Text-to-Text LLMs
General language models that can perform reranking:

```python
# Amazon Nova family
RerankingConfig(
    model_type="llm",
    model_id="us.amazon.nova-lite-v1:0",
    model_parameters={"temperature": 0.1, "max_tokens": 1000}
)

# Anthropic Claude family
RerankingConfig(
    model_type="llm",
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    model_parameters={"temperature": 0.1, "max_tokens": 1000}
)
```

## PDF Processing

PDF processing uses multimodal LLMs that can understand both text and images. These models analyze images within PDFs to extract text content and describe visual elements.

### Supported Multimodal Models
- **Nova Lite**: Amazon's multimodal model
- **Nova Pro**: Higher quality image analysis
- **Claude Sonnet 4**: Anthropic's multimodal model
- Other models available on Bedrock may work, but have not been tested

### PDF Extractor Configuration
```python
from multi_class_text_classifier import create_pdf_extractor

pdf_extractor = create_pdf_extractor(
    model_id="us.amazon.nova-lite-v1:0"
)

pdf_content = pdf_extractor.extract_from_file("document.pdf")
```

## Dataset Generation

Dataset generation creates synthetic class definitions for your domain using LLMs. This is useful when you don't have existing class definitions but need to classify text into domain-specific categories. The generated classes include realistic names and descriptions that capture the nuances of your domain.

```python
generator = DatasetGenerator()

# Technology domain
tech_dataset = generator.generate_dummy_dataset("software tools", 50)

# Healthcare domain  
health_dataset = generator.generate_dummy_dataset("medical devices", 75)

# E-commerce domain
ecommerce_dataset = generator.generate_dummy_dataset("product categories", 100)
```

## Understanding Results

### Score Types
- **Similarity Score**: Cosine similarity from embedding search
- **Rerank Score**: Semantic evaluation from LLM models
- **Effective Score**: Best available score (rerank if available, otherwise similarity)

### Accessing Detailed Results
```python
result = classifier.predict("Sample text")

print(f"Predicted: {result.predicted_class.name}")
print(f"Confidence: {result.effective_score:.4f} ({result.get_confidence_level()})")

for alt in result.alternatives[1:3]:
    print(f"Alternative: {alt.class_definition.name} ({alt.effective_score:.4f})")
```

## Attribute Validation

Attribute validation provides an additional layer of confidence by checking if the classified text meets specific conditions defined for each class. This helps identify misclassifications and provides detailed feedback.

### Example in Practice

If a document is classified as an **Invoice**, it must have:
- A monetary amount to be paid (e.g., "$500.00", "Total: $1,250")
- Both parties identified (e.g., "From: ABC Corp" and "To: XYZ Inc")
- Either a request for payment OR proof of completed payment

If classified as a **Technical Manual**, it must have:
- Instructions for operating or maintaining something
- Step-by-step procedures (numbered or bulleted lists)
- Technical specifications, requirements, or interface descriptions

This validation catches cases where similarity search might classify a receipt as an invoice, or a blog post as a technical manual.

### How It Works

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Predicted Class │───▶│ Load Attributes │───▶│ LLM Evaluation  │───▶│ Logical Scoring │
│                 │    │                 │    │                 │    │                 │
│ "Invoice"       │    │ Load conditions │    │ Check each      │    │ Apply AND/OR    │
│ Score: 0.85     │    │ for Invoice     │    │ condition       │    │ logic           │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                ↓                       ↓                       ↓
                        JSON Definitions        Individual Results      Final Attribute Score
                        • Amount required       • Amount: ✓ YES         Score: 1.0 (all met)
                        • Parties required      • Parties: ✓ YES        Conditions Met: 3
                        • Payment type          • Payment: ✓ YES        Conditions Not Met: 0
```

### Attribute Score Types
- **Attribute Score**: 0.0-1.0 based on how well the text matches class-specific conditions
- **Combined Confidence**: Average of similarity and attribute scores for comprehensive assessment
- **Conditions Met/Not Met**: Detailed breakdown of which conditions passed or failed with reasoning

### Example Attribute Definition
```json
{
  "name": "Invoice",
  "required_attributes": {
    "operator": "AND",
    "conditions": [
      "must specify monetary amount to be paid",
      "must identify both parties (payer and payee)",
      {
        "operator": "OR", 
        "conditions": [
          "must be a request for payment",
          "must be a proof of completed payment"
        ]
      }
    ]
  }
}
```

## Examples

The `examples/getting_started/` directory contains step-by-step examples:

### Basic Classification Examples
1. **Basic Classification** - Simple text classification using pre-built datasets
2. **Classification with Reranking** - Improved accuracy with LLM reranking
3. **PDF Document Classification** - Extract and classify PDF content using multimodal LLMs

### Attribute Validation Examples
4a. **Simple Attribute Validation** - Enhanced classification confidence with predefined attributes
4c. **Generate Attributes with LLM** - Automatically create attribute definitions using LLMs
4d. **Use Generated Attributes** - Complete workflow from generation to practical usage

### Advanced Examples
5a. **Complete Pipeline** - Full workflow with dataset generation, embeddings, and classification
5b. **Reranking Comparison** - Compare different reranking models and their performance

## Use Cases

- **Document Classification**: Invoices, contracts, reports, manuals
- **Product Classification**: E-commerce, inventory, parts catalogs
- **Content Categorization**: Articles, support tickets, emails
- **Medical Classification**: Supplies, equipment, procedures
- **Technical Classification**: Software tools, hardware components

## Performance Tips

1. **Use existing embeddings** when possible for faster initialization
2. **Limit top_k_candidates** (3-10) for optimal reranking performance
3. **Batch process** multiple texts for better throughput

## Project Structure

```
multi_class_text_classifier/
├── multi_class_text_classifier/     # Core package
├── examples/getting_started/        # Step-by-step examples
├── datasets/                        # Sample datasets
├── tests/                          # Unit tests
└── output/                         # Generated files
```

The binary files `datasets/*_embeddings.pkl.gz` contain the embeddings of the
class descriptions contained in the corresponding json file. These are
generated automatically when the dataset is loaded, but are are provided in the
repository to speed up initialization.

## Third-Party Dependencies and Licenses

This package depends on and may incorporate or retrieve a number of third-party
software packages (such as open source packages) at install-time or build-time
or run-time ("External Dependencies"). The External Dependencies are subject to
license terms that you must accept in order to use this package. If you do not
accept all of the applicable license terms, you should not use this package. We
recommend that you consult your company’s open source approval policy before
proceeding.

Provided below is a list of External Dependencies and the applicable license
identification as indicated by the documentation associated with the External
Dependencies as of Amazon's most recent review.

THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT PROMISE THAT
THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE COMPLETE, ACCURATE, OR
UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR ANY INACCURACIES. YOU SHOULD
CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL DEPENDENCIES FOR THE MOST COMPLETE
AND UP-TO-DATE LICENSING INFORMATION.

YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT WILL
AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT,
INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE DAMAGES (INCLUDING
FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, OR
COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR RELATING TO THE EXTERNAL
DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, EVEN
IF AMAZON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS
AND DISCLAIMERS APPLY EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

### AGPL Dependencies

- **PyMuPDF** (1.26.3) - GNU Affero General Public License v3.0 (AGPL-3.0) - https://github.com/pymupdf/PyMuPDF

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.