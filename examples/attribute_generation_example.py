"""
Example demonstrating how to use AttributeGenerator for automatic attribute generation.
"""

from multi_class_text_classifier import AttributeGenerator
from multi_class_text_classifier.models.data_models import (
    AttributeGenerationConfig,
    ClassDefinition
)
import json


def main():
    """Demonstrate AttributeGenerator usage."""
    print("🔧 AttributeGenerator Example")
    print("=" * 50)
    
    try:
        # 1. Create configuration for Claude Sonnet 4
        config = AttributeGenerationConfig(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            temperature=0.1,
            max_tokens=4000
        )
        print(f"✅ Created configuration with model: {config.model_id}")
        
        # 2. Initialize AttributeGenerator
        generator = AttributeGenerator(config)
        print("✅ Initialized AttributeGenerator")
        
        # 3. Define sample classes to generate attributes for
        sample_classes = [
            ClassDefinition(
                name="Research Paper",
                description="Academic papers, research studies, white papers, scientific publications, and scholarly articles with research findings, methodologies, and citations."
            ),
            ClassDefinition(
                name="Legal Contract",
                description="Legal agreements, contracts, terms of service, privacy policies, licensing agreements, and regulatory compliance documents with legal terms and conditions."
            ),
            ClassDefinition(
                name="Marketing Material",
                description="Promotional content, brochures, advertisements, marketing communications, product catalogs, and sales materials designed to promote products or services."
            )
        ]
        
        # 4. Generate attributes for each class
        generated_attributes = {}
        
        for class_def in sample_classes:
            print(f"\n🎯 Generating attributes for: {class_def.name}")
            print(f"Description: {class_def.description[:100]}...")
            
            try:
                # Generate attributes with domain context
                result = generator.generate_attributes_for_class(
                    class_def,
                    domain_context="document classification"
                )
                
                generated_attributes[class_def.name] = result
                
                # Display results
                print(f"✅ Generated attributes successfully!")
                print(f"   Operator: {result['required_attributes']['operator']}")
                print(f"   Conditions: {len(result['required_attributes']['conditions'])}")
                
                # Show first few conditions
                conditions = result['required_attributes']['conditions']
                for i, condition in enumerate(conditions[:2]):
                    if isinstance(condition, str):
                        print(f"   - {condition}")
                    else:
                        print(f"   - {condition['operator']} with {len(condition['conditions'])} sub-conditions")
                
                if len(conditions) > 2:
                    print(f"   ... and {len(conditions) - 2} more conditions")
                
            except Exception as e:
                print(f"❌ Failed to generate attributes: {e}")
                continue
        
        # 5. Save results to file
        if generated_attributes:
            output_file = "generated_attributes.json"
            
            # Create output structure similar to existing attributes file
            output_data = {
                "metadata": {
                    "domain": "document types",
                    "num_classes": len(generated_attributes),
                    "generated_by": "AttributeGenerator",
                    "model": config.model_id,
                    "version": "1.0"
                },
                "classes": []
            }
            
            for class_name, attrs in generated_attributes.items():
                output_data["classes"].append(attrs)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Saved generated attributes to: {output_file}")
            print(f"📊 Generated attributes for {len(generated_attributes)} classes")
        
        print("\n🎉 AttributeGenerator example completed successfully!")
        
    except ImportError as e:
        if "strands" in str(e).lower():
            print("❌ Strands Agents SDK not available.")
            print("   Please install and configure Strands Agents to use AttributeGenerator.")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()