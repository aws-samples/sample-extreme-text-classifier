"""
Classify using existing embeddings - much faster since embeddings are already generated.
This shows that you only need the embeddings path once they've been created.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import TextClassifier

print("Medical Supplies Classification (Using Existing Embeddings)")
print("=" * 60)

# Initialize classifier with existing embeddings
# This is much faster since embeddings are already generated
print("Initializing classifier with existing embeddings...")
classifier = TextClassifier(
    embeddings_path="output/medical_supplies_embeddings.pkl.gz"
)

print("✓ Classifier initialized quickly using existing embeddings")

# Test with different medical supplies text
test_texts = [
    "Oxygen concentrator for respiratory therapy at home",
    "Surgical scissors with curved blades for precise cutting",
    "Electronic thermometer with digital display for temperature measurement",
    "Wheelchair with adjustable footrests for patient mobility"
]

print(f"\nClassifying {len(test_texts)} new medical supply descriptions:")
print("-" * 70)

for i, text in enumerate(test_texts, 1):
    result = classifier.predict(text)
    print(f"\n{i}. Text: {text}")
    print(f"   → Classified as: {result.predicted_class.name}")
    print(f"   → Confidence: {result.effective_score:.4f} ({result.get_confidence_level()})")
    print(f"   → Description: {result.predicted_class.description[:60]}...")

print(f"\n💡 Key Point: Once embeddings are generated, you can:")
print(f"   • Initialize the classifier much faster")
print(f"   • Only need the embeddings file path")
print(f"   • Embeddings contain the vector representations of all classes")