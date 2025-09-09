"""
Use the generated medical supplies dataset for classification.
This will generate embeddings for the first time.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import TextClassifier

print("Medical Supplies Classification")
print("=" * 50)

# Initialize classifier with medical supplies dataset
# This will generate embeddings for the first time
print("Initializing classifier (generating embeddings for the first time)...")
classifier = TextClassifier(
    dataset_path="output/medical_supplies_dataset.json",
    embeddings_path="output/medical_supplies_embeddings.pkl.gz"
)

print("✓ Classifier initialized and embeddings generated")

# Test medical supplies classification
test_texts = [
    "Disposable surgical gloves made of latex for medical procedures",
    "Digital blood pressure monitor with automatic cuff inflation",
    "Sterile gauze pads for wound dressing and bandaging",
    "Stethoscope with dual-head chest piece for cardiac examination"
]

print(f"\nClassifying {len(test_texts)} medical supply descriptions:")
print("-" * 60)

for i, text in enumerate(test_texts, 1):
    result = classifier.predict(text)
    print(f"\n{i}. Text: {text}")
    print(f"   → Classified as: {result.predicted_class.name}")
    print(f"   → Confidence: {result.effective_score:.4f} ({result.get_confidence_level()})")
    
    # Show top alternative
    if len(result.alternatives) > 1:
        alt = result.alternatives[1]  # Second best
        print(f"   → Alternative: {alt.class_definition.name} ({alt.effective_score:.4f})")

print(f"\n✓ Embeddings saved to output/medical_supplies_embeddings.pkl.gz")
print("Next: Run 4c_classify_with_existing_embeddings.py to see how to reuse embeddings")