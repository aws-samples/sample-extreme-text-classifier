"""
Simple attribute validation example with a single test case.

This example shows the basic usage of attribute validation in the classifier.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier import TextClassifier
from multi_class_text_classifier.models.data_models import (
    AttributeValidationConfig,
    AttributeEvaluationConfig
)

def main():
    """Simple attribute validation example."""
    
    print("🎯 Simple Attribute Validation Example")
    print("=" * 50)
    
    # Configure attribute validation
    attribute_config = AttributeValidationConfig(
        enabled=True,
        attributes_path="datasets/document_classification_attributes.json",
        evaluation_config=AttributeEvaluationConfig(
            model_id="us.amazon.nova-lite-v1:0",
            temperature=0.1,
            max_tokens=1000
        )
    )
    
    # Initialize classifier with attribute validation
    print("🔧 Initializing classifier with attribute validation...")
    classifier = TextClassifier(
        dataset_path="datasets/document_classification_dataset.json",
        embeddings_path="datasets/document_classification_embeddings.pkl.gz",
        attribute_config=attribute_config
    )
    print("✅ Classifier initialized!\n")
    
    # Test with a complete invoice
    test_text = """INVOICE #INV-2024-001

Bill To: ABC Corporation
123 Business St
New York, NY 10001

From: XYZ Services LLC
456 Service Ave
Boston, MA 02101

Amount Due: $2,500.00
Due Date: March 15, 2024

Please remit payment within 30 days of invoice date."""
    
    print("📝 Classifying text:")
    print(f"   {test_text[:80]}...")
    print()
    
    # Classify with attribute validation
    result = classifier.predict(test_text)
    
    # Display results
    print("🎯 Results:")
    print(f"   Predicted Class: {result.predicted_class.name}")
    print(f"   Similarity Score: {result.similarity_score:.3f}")
    
    if result.attribute_validation:
        attr_result = result.attribute_validation
        print(f"   Attribute Score: {attr_result.overall_score:.3f}")
        
        print(f"\n✅ Attribute Details:")
        if attr_result.conditions_met:
            print(f"   ✓ Conditions Met:")
            for condition in attr_result.conditions_met:
                print(f"     • {condition}")
        
        if attr_result.conditions_not_met:
            print(f"   ✗ Conditions Not Met:")
            for condition in attr_result.conditions_not_met:
                print(f"     • {condition}")
    else:
        print(f"   Attribute Score: N/A")
    
    print()

if __name__ == "__main__":
    main()