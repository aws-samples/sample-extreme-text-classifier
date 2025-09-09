"""
Attribute generation example demonstrating automatic attribute creation using Amazon Nova Pro.

This example shows how to generate attributes for document classification classes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from multi_class_text_classifier.attribute_generator import AttributeGenerator
from multi_class_text_classifier.dataset_loader import ClassesDataset
from multi_class_text_classifier.models.data_models import AttributeGenerationConfig

def display_generated_attributes(attributes: dict):
    """Display generated attributes in a readable format."""
    print(f"\n📋 Generated Attributes")
    print("=" * 50)
    
    metadata = attributes['metadata']
    print(f"📊 Generated {metadata['num_classes']} classes using {metadata['model']}")
    
    for i, class_data in enumerate(attributes['classes'], 1):
        try:
            print(f"\n🎯 Class {i}: {class_data['name']}")
            
            attrs = class_data['required_attributes']
            print(f"   Required Attributes ({attrs['operator']}):")
            
            for condition in attrs['conditions']:
                if isinstance(condition, str):
                    print(f"     • {condition}")
                elif isinstance(condition, dict) and 'operator' in condition:
                    print(f"     • ({condition['operator']}):")
                    for sub_condition in condition['conditions']:
                        if isinstance(sub_condition, str):
                            print(f"       - {sub_condition}")
                        elif isinstance(sub_condition, dict):
                            print(f"       - ({sub_condition.get('operator', 'N/A')}):")
                            for nested_condition in sub_condition.get('conditions', []):
                                print(f"         * {nested_condition}")
        except (KeyError, IndexError) as e:
            print(f"\n❌ Error displaying class {i}: {e}")
            print(f"   Raw data: {class_data}")

def main():
    """Main function demonstrating attribute generation."""
    
    print("🎯 Attribute Generation Example")
    print("=" * 50)
    print("Generating attributes for document classification classes using Amazon Nova Pro.\n")
    
    # Load existing classes from dataset
    dataset_path = "datasets/document_classification_dataset.json"
    print(f"📂 Loading classes from {dataset_path}...")
    
    dataset_loader = ClassesDataset(dataset_path)
    classes = dataset_loader.load_classes_from_json()
    print(f"✅ Loaded {len(classes)} classes")
    
    # Configure attribute generation with Amazon Nova Pro
    generation_config = AttributeGenerationConfig(
        model_id="us.amazon.nova-pro-v1:0",
        temperature=0.1,
        max_tokens=4000
    )
    
    # Initialize attribute generator
    print("🔧 Initializing AttributeGenerator...")
    generator = AttributeGenerator(generation_config)
    print("✅ AttributeGenerator initialized!\n")
    
    # Generate attributes for all classes
    print(f"🔄 Generating attributes for all {len(classes)} classes...")
    
    all_attributes = generator.generate_attributes_for_classes(
        classes,
        domain_context="document classification"
    )
    print(f"✅ Generated attributes for all classes!")
    
    # Attributes are already in standard format
    standard_attributes = all_attributes
    
    # Save to output folder
    output_path = "output/generated_document_classification_attributes.json"
    print(f"💾 Saving to {output_path}...")
    
    generator.save_attributes_to_file(standard_attributes, output_path)
    print(f"✅ Saved successfully!")
    
    # Display the results
    display_generated_attributes(standard_attributes)
    
    print(f"\n🎉 Attribute generation completed!")
    print(f"Generated attributes saved to: {output_path}")

if __name__ == "__main__":
    main()