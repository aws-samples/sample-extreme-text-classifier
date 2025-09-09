"""
Comprehensive attribute validation example with multiple test cases.

This example demonstrates attribute validation across different document types
and shows how attribute scores provide additional confidence signals.
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
    """Main function demonstrating comprehensive attribute validation."""
    
    print("🎯 Comprehensive Attribute Validation Example")
    print("=" * 60)
    print("This example shows attribute validation across different document types")
    print("and demonstrates how attribute scores enhance classification confidence.\n")
    
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
    
    # Initialize classifier with attribute validation enabled
    print("🔧 Initializing classifier with attribute validation...")
    classifier = TextClassifier(
        dataset_path="datasets/document_classification_dataset.json",
        embeddings_path="datasets/document_classification_embeddings.pkl.gz",
        attribute_config=attribute_config
    )
    print("✅ Classifier initialized successfully!\n")
    
    # Test texts that should match different document types with varying attribute compliance
    test_cases = [
        {
            "text": "INVOICE #INV-2024-001\n\nBill To: ABC Corporation\n123 Business St\nNew York, NY 10001\n\nFrom: XYZ Services LLC\n456 Service Ave\nBoston, MA 02101\n\nAmount Due: $2,500.00\nDue Date: March 15, 2024\n\nPlease remit payment within 30 days of invoice date.",
            "description": "Complete invoice with all required attributes"
        },
        {
            "text": "Payment receipt for services rendered. Amount: $500. Thank you for your business.",
            "description": "Partial invoice - missing some required attributes"
        },
        {
            "text": "USER MANUAL\n\nChapter 1: Installation\n1. Download the software package\n2. Run the installer\n3. Follow the setup wizard\n\nChapter 2: Configuration\n1. Open the application\n2. Navigate to Settings\n3. Configure your preferences\n\nThis manual covers all technical specifications and requirements.",
            "description": "Technical manual with clear instructions"
        },
        {
            "text": "This document contains some instructions but lacks clear step-by-step procedures and technical details.",
            "description": "Weak technical manual - missing key attributes"
        },
        {
            "text": "LEGAL AGREEMENT\n\nThis contract is entered into between Party A (Client) and Party B (Service Provider).\n\nTerms and Conditions:\n1. Service Provider agrees to deliver services as specified\n2. Client agrees to pay fees as outlined in Schedule A\n3. This agreement is binding upon both parties\n\nEffective Date: January 1, 2024\nSignatures required from both parties.",
            "description": "Complete legal contract"
        },
        {
            "text": "Some legal text mentioning agreements but without clear parties or binding terms.",
            "description": "Weak legal document - missing key attributes"
        }
    ]
    
    print("📋 Classifying test cases with attribute validation:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"{'='*60}")
        print(f"📝 Text: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
        print()
        
        try:
            # Classify with attribute validation
            result = classifier.predict(test_case['text'], top_k=3)
            
            # Display classification results
            print(f"🎯 Classification Results:")
            print(f"   Predicted Class: {result.predicted_class.name}")
            print(f"   Similarity Score: {result.similarity_score:.3f}")
            
            # Display attribute validation results
            if result.attribute_validation:
                attr_result = result.attribute_validation
                print(f"   Attribute Score: {attr_result.overall_score:.3f}")
                print(f"   Combined Confidence: {(result.similarity_score + attr_result.overall_score) / 2:.3f}")
                
                print(f"\n✅ Attribute Validation Details:")
                if attr_result.conditions_met:
                    print(f"   ✓ Conditions Met ({len(attr_result.conditions_met)}):")
                    for condition in attr_result.conditions_met:
                        print(f"     • {condition}")
                
                if attr_result.conditions_not_met:
                    print(f"   ✗ Conditions Not Met ({len(attr_result.conditions_not_met)}):")
                    for condition in attr_result.conditions_not_met:
                        print(f"     • {condition}")
                
                if not attr_result.conditions_met and not attr_result.conditions_not_met:
                    print(f"   ⚠️  No specific conditions evaluated")
                
                # Show evaluation details if available
                if attr_result.evaluation_details.get('reasoning'):
                    print(f"   💭 Reasoning: {attr_result.evaluation_details['reasoning']}")
            
            else:
                print(f"   Attribute Score: N/A (no validation performed)")
                if result.metadata.get('attribute_validation_skipped'):
                    print(f"   ⚠️  {result.metadata.get('attribute_validation_reason', 'Attribute validation skipped')}")
                elif result.metadata.get('attribute_validation_error'):
                    print(f"   ❌ Attribute validation error: {result.metadata['attribute_validation_error']}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error classifying text: {str(e)}")
            print()
    
    # Demonstrate the impact of attribute validation on confidence
    print(f"{'='*60}")
    print("📊 Impact of Attribute Validation on Confidence")
    print(f"{'='*60}")
    print("Attribute validation provides an additional confidence signal that can:")
    print("• ✅ Increase confidence when attributes match well")
    print("• ⚠️  Decrease confidence when attributes don't match")
    print("• 🎯 Help identify misclassifications even with high similarity scores")
    print("• 📈 Provide detailed feedback on why a classification succeeded or failed")
    print()

if __name__ == "__main__":
    main()