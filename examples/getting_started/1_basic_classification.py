"""
Basic text classification example using document classification dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import TextClassifier

# Initialize classifier with document classification dataset
# If embeddings file does not exist, it will generate embeddings from json file and create it
classifier = TextClassifier(
    dataset_path="datasets/document_classification_dataset.json",
    embeddings_path="datasets/document_classification_embeddings.pkl.gz"
)

# Classify different types of text
test_texts = [
    "Please remit payment of $1,250.00 within 30 days of invoice date. Payment terms: Net 30.",
    "This user manual provides step-by-step instructions for installing and configuring the software application.",
    "The parties agree to the terms and conditions set forth in this agreement, effective as of the date of execution."
]

print("Document Classification Results:")
print("=" * 50)

for i, text in enumerate(test_texts, 1):
    result = classifier.predict(text)
    print(f"\n{i}. Text: {text[:60]}...")
    print(f"   Predicted Class: {result.predicted_class.name}")
    print(f"   Confidence: {result.effective_score:.4f}")
    
    # Show top 2 alternatives
    print(f"   Top alternatives:")
    for alt in result.alternatives[1:3]:
        print(f"     - {alt.class_definition.name}: {alt.effective_score:.4f}")