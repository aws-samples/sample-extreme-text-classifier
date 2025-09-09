"""
PDF document classification - extract content from PDF and classify it.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import (
    create_pdf_extractor,
    TextClassifier
)

print("PDF Document Classification")
print("=" * 50)

# Create PDF extractor and specify which LLM to use
pdf_extractor = create_pdf_extractor(
    model_id="us.amazon.nova-lite-v1:0"
)

# Extract content from existing invoice PDF
print("Extracting content from PDF...")
pdf_content = pdf_extractor.extract_from_file("datasets/dummy_invoice.pdf")

print(f"✓ Extracted {len(pdf_content.text_content)} characters of text")
print(f"✓ Found {len(pdf_content.images)} images")

# Show content preview
if pdf_content.full_content:
    preview = pdf_content.full_content[:200] + "..." if len(pdf_content.full_content) > 200 else pdf_content.full_content
    print(f"\nContent preview: {preview}")

# Initialize classifier with reranking - only need embeddings path since dataset was already processed in previous example
classifier = TextClassifier(
    embeddings_path="datasets/document_classification_embeddings.pkl.gz"
)

# Classify the extracted PDF content
print("\nClassifying extracted content...")
result = classifier.predict(pdf_content.full_content)

print(f"\n🎯 Classification Result:")
print(f"Document type: {result.predicted_class.name}")
print(f"Confidence: {result.effective_score:.2%}")
print(f"Description: {result.predicted_class.description}")

# Show top alternatives
print(f"\nTop alternatives:")
for alt in result.alternatives[1:3]:
    print(f"  {alt.class_definition.name}: {alt.effective_score:.2%}")