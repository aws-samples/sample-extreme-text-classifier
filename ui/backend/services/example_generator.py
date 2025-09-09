"""
Example generator service for creating realistic document examples using AWS Bedrock.
"""

import logging
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import config, ExampleGenerationPrompts, ToolDefinitions, SystemPrompts

logger = logging.getLogger(__name__)


class ExampleGenerationError(Exception):
    """Raised when example generation fails."""
    pass


class ExampleGenerator:
    """Generates realistic document examples using Amazon Nova Pro."""
    
    def __init__(self, region_name: str = None):
        """Initialize the example generator with AWS Bedrock client."""
        self.region_name = region_name or config.aws.bedrock_region
        self.model_id = config.aws.default_nova_pro_model
        
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region_name)
            logger.info(f"Initialized ExampleGenerator with model: {self.model_id} in region: {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_client = None
    

    
    def generate_examples(
        self, 
        classes: List[Dict[str, str]], 
        domain: str, 
        num_examples_per_class: int = 1
    ) -> Dict[str, List[str]]:
        """
        Generate realistic examples for multiple classes using batch processing and tool use.
        
        Args:
            classes: List of class dictionaries with 'name' and 'description' keys
            domain: Domain context (e.g., "medical supplies", "legal documents")
            num_examples_per_class: Number of examples to generate per class
            
        Returns:
            Dictionary mapping class names to lists of generated examples
        """
        if not self.bedrock_client:
            logger.error("Bedrock client not available, using fallback examples")
            return self._create_fallback_examples_all(classes, domain, num_examples_per_class)
        
        if not classes:
            return {}
        
        try:
            prompt = ExampleGenerationPrompts.batch_example_generation_prompt(
                classes, domain, num_examples_per_class
            )
            
            logger.info(f"Generating {num_examples_per_class} example(s) for {len(classes)} classes in batch")
            
            # Define the tool for structured output
            tool_definition = ToolDefinitions.generate_examples_tool()
            
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "temperature": config.models.example_generation_temperature,
                    "maxTokens": config.models.example_generation_max_tokens
                },
                toolConfig={
                    "tools": [tool_definition],
                    "toolChoice": {"tool": {"name": "generate_examples"}}
                }
            )
            
            # Extract structured output from tool use
            if 'toolUse' in response['output']['message']['content'][0]:
                tool_use = response['output']['message']['content'][0]['toolUse']
                examples_data = tool_use['input']['examples']
                
                # Validate and clean the results
                results = self._validate_batch_examples(examples_data, classes, domain, num_examples_per_class)
                
                logger.info(f"Successfully generated examples for {len(results)} classes using tool use")
                return results
            else:
                logger.warning("No tool use found in response, falling back to text parsing")
                # Fallback to text parsing if tool use fails
                generated_text = response['output']['message']['content'][0]['text']
                return self._parse_batch_examples_from_text(generated_text, classes, domain, num_examples_per_class)
                
        except Exception as e:
            logger.error(f"Batch example generation failed: {e}")
            return self._create_fallback_examples_all(classes, domain, num_examples_per_class)
    

    

    
    def _create_fallback_examples(self, class_name: str, domain: str, count: int) -> List[str]:
        """Create simple fallback examples when generation fails."""
        return [
            SystemPrompts.fallback_example_template(class_name, domain, i+1)
            for i in range(count)
        ]
    
    def _create_fallback_examples_all(
        self, 
        classes: List[Dict[str, str]], 
        domain: str, 
        num_examples_per_class: int
    ) -> Dict[str, List[str]]:
        """Create fallback examples for all classes."""
        results = {}
        for cls in classes:
            class_name = cls.get('name', 'Unknown')
            results[class_name] = self._create_fallback_examples(class_name, domain, num_examples_per_class)
        return results
    
    def _validate_batch_examples(
        self, 
        examples_data: Dict[str, List[str]], 
        classes: List[Dict[str, str]], 
        domain: str, 
        num_examples_per_class: int
    ) -> Dict[str, List[str]]:
        """Validate and clean batch-generated examples."""
        results = {}
        
        for cls in classes:
            class_name = cls.get('name', 'Unknown')
            
            if class_name in examples_data:
                examples = examples_data[class_name]
                
                # Validate examples
                valid_examples = []
                for example in examples:
                    if isinstance(example, str) and len(example.strip()) > 10:
                        valid_examples.append(example.strip())
                
                # Ensure we have the right number of examples
                while len(valid_examples) < num_examples_per_class:
                    valid_examples.append(SystemPrompts.professional_fallback_template(class_name, domain))
                
                results[class_name] = valid_examples[:num_examples_per_class]
            else:
                # Class not found in generated examples, create fallback
                results[class_name] = self._create_fallback_examples(class_name, domain, num_examples_per_class)
        
        return results
    
    def _parse_batch_examples_from_text(
        self, 
        generated_text: str, 
        classes: List[Dict[str, str]], 
        domain: str, 
        num_examples_per_class: int
    ) -> Dict[str, List[str]]:
        """Fallback text parsing for batch examples."""
        logger.info("Using fallback text parsing for batch examples")
        
        results = {}
        lines = generated_text.strip().split('\n')
        current_class = None
        current_examples = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line indicates a new class
            for cls in classes:
                class_name = cls['name']
                if class_name.lower() in line.lower() and ':' in line:
                    # Save previous class examples
                    if current_class and current_examples:
                        results[current_class] = current_examples[:num_examples_per_class]
                    
                    current_class = class_name
                    current_examples = []
                    break
            else:
                # This is an example line
                if current_class and (line[0].isdigit() or line.startswith('-')):
                    # Clean up the example
                    if '.' in line and line[0].isdigit():
                        example = line.split('.', 1)[1].strip()
                    elif line.startswith('-'):
                        example = line[1:].strip()
                    else:
                        example = line
                    
                    if example and len(example) > 10:
                        current_examples.append(example)
        
        # Save the last class
        if current_class and current_examples:
            results[current_class] = current_examples[:num_examples_per_class]
        
        # Fill in missing classes with fallbacks
        for cls in classes:
            class_name = cls['name']
            if class_name not in results:
                results[class_name] = self._create_fallback_examples(class_name, domain, num_examples_per_class)
        
        return results
    
    def test_connection(self) -> bool:
        """Test the connection to AWS Bedrock."""
        if not self.bedrock_client:
            return False
        
        try:
            # Simple test with minimal prompt
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": ExampleGenerationPrompts.connection_test_prompt()}]
                    }
                ],
                inferenceConfig={
                    "temperature": config.models.default_temperature,
                    "maxTokens": 50
                }
            )
            
            response_text = response['output']['message']['content'][0]['text']
            return "successful" in response_text.lower()
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Global instance for use in the API
example_generator = ExampleGenerator()