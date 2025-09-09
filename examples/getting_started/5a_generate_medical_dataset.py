"""
Generate a medical supplies dataset for classification.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import DatasetGenerator

print("Generating Medical Supplies Dataset")
print("=" * 50)

# Generate a dataset for medical supplies domain
generator = DatasetGenerator()
print("Generating dataset... (this may take a moment)")

dataset = generator.generate_dummy_dataset(
    domain="medical supplies",
    num_classes=20
)

# Save to datasets folder
dataset_path = "output/medical_supplies_dataset.json"
generator.save_dataset(dataset, dataset_path)

print(f"✓ Dataset generated and saved to {dataset_path}")
print(f"✓ Generated {len(dataset['classes'])} classes for medical supplies domain")

# Show some example classes
print(f"\nExample classes generated:")
for i, class_info in enumerate(dataset['classes'][:5], 1):
    print(f"{i}. {class_info['name']}")
    print(f"   {class_info['description'][:80]}...")
    print()

print("Next: Run 4b_classify_medical_supplies.py to use this dataset for classification")