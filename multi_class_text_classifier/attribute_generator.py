"""
Automatic attribute generator for classification classes using AWS Bedrock.
"""

import json
import logging
import boto3
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from .models.data_models import ClassDefinition, AttributeGenerationConfig
from .config import config
from .prompts import AttributeGenerationPrompts, ToolDefinitions


logger = logging.getLogger(__name__)


class AttributeGenerator:
    """
    Generates attribute definitions automatically using AWS Bedrock.
    """
    
    def __init__(self, model_config: AttributeGenerationConfig):
        """
        Initialize attribute generator with model configuration.
        
        Args:
            model_config: Configuration for the model
        """
        self.model_config = model_config
        self._bedrock_client = None
        
    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client."""
        if self._bedrock_client is None:
            try:
                # Initialize Bedrock client with debugging
                region = config.aws.bedrock_region
                logger.info(f"Initializing Bedrock client in region: {region}")
                
                # Configure retry behavior optimized for Bedrock throttling
                from botocore.config import Config
                
                retry_config = Config(
                    retries={
                        'total_max_attempts': config.retry.max_attempts,
                        'mode': config.retry.mode
                    },
                    read_timeout=config.retry.read_timeout,
                    connect_timeout=config.retry.connect_timeout,
                    max_pool_connections=config.retry.max_pool_connections
                )
                
                self._bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=region,
                    config=retry_config
                )
                
                # Log detailed client configuration
                logger.info(f"Bedrock client configured for region: {region}")
                logger.info(f"Client region: {self._bedrock_client.meta.region_name}")
                logger.info(f"Endpoint URL: {self._bedrock_client.meta.endpoint_url}")
                
                # Also check environment variables that might override region
                logger.info(f"AWS_DEFAULT_REGION env: {os.environ.get('AWS_DEFAULT_REGION', 'Not set')}")
                logger.info(f"AWS_REGION env: {os.environ.get('AWS_REGION', 'Not set')}")
                
                # Test the actual endpoint being used
                try:
                    # This will show us the actual endpoint being called
                    logger.info(f"Testing endpoint resolution...")
                    test_params = {
                        'modelId': 'us.amazon.nova-pro-v1:0',
                        'messages': [{'role': 'user', 'content': [{'text': 'test'}]}],
                        'inferenceConfig': {'maxTokens': 1}
                    }
                    # Don't actually call, just prepare the request to see the endpoint
                    request = self._bedrock_client._make_request(
                        operation_model=self._bedrock_client._service_model.operation_model('Converse'),
                        request_dict=self._bedrock_client._build_request(test_params, 'Converse')
                    )
                except Exception as e:
                    logger.debug(f"Endpoint test failed (expected): {e}")
                    # This is expected to fail, but it will show us the endpoint in debug logs
                
                # Test the client by listing available models (optional debug step)
                logger.info(f"Bedrock client initialized successfully with model: {self.model_config.model_id}")
                logger.info(f"AWS credentials available: {self._bedrock_client._client_config.region_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                logger.error(f"AWS credentials check - Region: {os.environ.get('AWS_DEFAULT_REGION', 'Not set')}")
                logger.error(f"AWS credentials check - Access Key: {'Set' if os.environ.get('AWS_ACCESS_KEY_ID') else 'Not set'}")
                raise RuntimeError(f"Failed to initialize Bedrock client: {e}") from e
    
    def generate_attributes_for_class(
        self, 
        class_definition: ClassDefinition,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate attribute definition for a single class.
        
        Args:
            class_definition: Class to generate attributes for
            domain_context: Optional domain context to guide generation
            
        Returns:
            Attribute definition dictionary for the class
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM generation fails
        """
        # Validate inputs
        if not class_definition:
            raise ValueError("Class definition cannot be None")
        if not class_definition.name or not class_definition.name.strip():
            raise ValueError("Class name cannot be empty")
        if not class_definition.description or not class_definition.description.strip():
            raise ValueError("Class description cannot be empty")
        
        # Initialize Bedrock client if needed
        self._initialize_bedrock_client()
        
        try:
            logger.debug(f"Generating attributes for class '{class_definition.name}' with model {self.model_config.model_id}")
            
            # Create prompt for single class attribute generation
            prompt = AttributeGenerationPrompts.single_class_generation_prompt(
                class_definition.name,
                class_definition.description,
                domain_context
            )
            attribute_tool = ToolDefinitions.generate_class_attributes_tool()
            
            # Call Bedrock with Converse API
            logger.info(f"Making Bedrock converse call for class '{class_definition.name}'")
            logger.debug(f"Model ID: {self.model_config.model_id}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            try:
                response = self._bedrock_client.converse(
                    modelId=self.model_config.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    toolConfig={
                        "tools": [attribute_tool]
                    },
                    inferenceConfig={
                        "temperature": getattr(self.model_config, 'temperature', 0.1),
                        "maxTokens": getattr(self.model_config, 'max_tokens', 4000)
                    }
                )
                logger.info(f"Bedrock converse call successful for class '{class_definition.name}'")
                logger.debug(f"Response keys: {list(response.keys())}")
                
                # Log retry information if available
                response_metadata = response.get('ResponseMetadata', {})
                retry_attempts = response_metadata.get('RetryAttempts', 0)
                if retry_attempts > 0:
                    logger.info(f"Bedrock call for '{class_definition.name}' succeeded after {retry_attempts} retries")
                
            except Exception as bedrock_error:
                logger.error(f"Bedrock converse call failed for class '{class_definition.name}': {bedrock_error}")
                logger.error(f"Error type: {type(bedrock_error)}")
                
                # Check if it's a throttling error
                error_code = getattr(bedrock_error, 'response', {}).get('Error', {}).get('Code', '')
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    logger.warning(f"Throttling detected for class '{class_definition.name}' - retries will handle this")
                
                # Log if this was after retries
                if hasattr(bedrock_error, 'response'):
                    retry_attempts = bedrock_error.response.get('ResponseMetadata', {}).get('RetryAttempts', 0)
                    if retry_attempts > 0:
                        logger.error(f"Failed after {retry_attempts} retry attempts")
                
                raise
            
            # Extract tool use result
            attribute_definition = self._extract_bedrock_tool_result(response, class_definition.name)
            
            logger.info(f"Successfully generated attributes for class '{class_definition.name}'")
            return attribute_definition
            
        except Exception as e:
            logger.error(f"Attribute generation failed for class '{class_definition.name}': {e}")
            raise RuntimeError(f"Failed to generate attributes: {e}") from e
    
    def _create_single_class_generation_prompt(
        self, 
        class_definition: ClassDefinition, 
        domain_context: Optional[str] = None
    ) -> str:
        """
        Create prompt for generating attributes for a single class.
        
        Args:
            class_definition: Class to generate attributes for
            domain_context: Optional domain context
            
        Returns:
            Formatted prompt string
        """
        domain_info = f"\nDomain Context: {domain_context}" if domain_context else ""
        
        prompt = f"""You are an expert in document classification. Generate logical attribute requirements for this document class.

Class: {class_definition.name}
Description: {class_definition.description}{domain_info}

INSTRUCTIONS:
1. Create specific, measurable attributes that define what makes a document belong to this class
2. Focus on the minimal required set of attributes to avoid classification errors
3. Make conditions specific enough to distinguish this class from others
4. Use clear, actionable language for each condition

REQUIREMENTS:
- Start with an AND operator at the root level
- Include 2-4 main conditions under the root level
- Use OR operators at second level only if needed, don't use AND operators except for root level
- Each condition should be a clear, specific requirement

EXAMPLE OUTPUT:
For a "Technical Manual" class, the output should look like:
{{
  "name": "Technical Manual",
  "description": "Technical documentation, user guides, instruction manuals with procedures and specifications",
  "required_attributes": {{
    "operator": "AND",
    "conditions": [
      "must contain instructions for operating or maintaining something",
      "must be structured as step-by-step procedures",
      {{
        "operator": "OR",
        "conditions": [
          "must describe technical specifications",
          "must describe technical requirements",
          "must describe technical interfaces"
        ]
      }}
    ]
  }}
}}

Use the generate_class_attributes tool to provide your response with the proper JSON structure."""

        return prompt
    
    def _extract_bedrock_tool_result(self, response: Dict[str, Any], class_name: str) -> Dict[str, Any]:
        """
        Extract tool use result from Bedrock Converse API response.
        
        Args:
            response: Response from Bedrock Converse API
            class_name: Name of the class for error reporting
            
        Returns:
            Parsed attribute definition dictionary
            
        Raises:
            RuntimeError: If tool result extraction fails
        """
        try:
            # Extract from Bedrock Converse API response
            output = response.get('output', {})
            message = output.get('message', {})
            content = message.get('content', [])
            
            # Look for tool use in the content
            for content_block in content:
                if 'toolUse' in content_block:
                    tool_use = content_block['toolUse']
                    if tool_use.get('name') == 'generate_class_attributes':
                        tool_input = tool_use.get('input', {})
                        logger.debug(f"Extracted tool result for class '{class_name}': {tool_input}")
                        return tool_input
            
            # If no tool use found, try to parse text content as JSON
            for content_block in content:
                if 'text' in content_block:
                    text_content = content_block['text']
                    return self._parse_json_from_message(text_content, class_name)
            
            # If we get here, no valid content was found
            raise RuntimeError(f"No tool use or parseable content found in response for class '{class_name}'")
                
        except Exception as e:
            logger.error(f"Failed to extract Bedrock tool result for class '{class_name}': {e}")
            logger.debug(f"Response content: {response}")
            raise RuntimeError(f"Failed to parse Bedrock response: {e}") from e
    
    def _parse_json_from_message(self, message, class_name: str) -> Dict[str, Any]:
        """
        Parse JSON from a message string or dict (fallback method).
        
        Args:
            message: Message string or dict that may contain JSON
            class_name: Name of the class for error reporting
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            RuntimeError: If JSON parsing fails
        """
        try:
            # If message is already a dict, return it directly
            if isinstance(message, dict):
                return message
            
            # Convert to string if not already
            message = str(message).strip()
            
            # Look for JSON block markers
            if "```json" in message:
                start = message.find("```json") + 7
                end = message.find("```", start)
                if end != -1:
                    json_str = message[start:end].strip()
                    return json.loads(json_str)
            
            # Look for curly braces
            start = message.find("{")
            end = message.rfind("}") + 1
            if start != -1 and end > start:
                json_str = message[start:end]
                return json.loads(json_str)
            
            # Try parsing the entire message as JSON
            return json.loads(message)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from message for class '{class_name}': {e}")
            logger.debug(f"Message content: {message}")
            raise RuntimeError(f"Failed to parse JSON response: {e}") from e
    
    def generate_attributes_for_classes(
        self, 
        classes: List[ClassDefinition],
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate attribute definitions for multiple classes.
        
        This method always generates attributes one-by-one to avoid token limits
        and ensure reliability with large numbers of classes.
        
        Args:
            classes: List of class definitions to generate attributes for
            domain_context: Optional domain context to guide generation
            
        Returns:
            Dictionary with generated attribute definitions in standard JSON format
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM generation fails
        """
        # Validate inputs
        if not classes:
            raise ValueError("Classes list cannot be empty")
        
        for i, class_def in enumerate(classes):
            if not class_def:
                raise ValueError(f"Class definition at index {i} cannot be None")
            if not class_def.name or not class_def.name.strip():
                raise ValueError(f"Class name at index {i} cannot be empty")
            if not class_def.description or not class_def.description.strip():
                raise ValueError(f"Class description at index {i} cannot be empty")
        
        logger.info(f"Generating attributes for {len(classes)} classes using iterative approach")
        
        try:
            # Generate classes iteratively with retry logic
            class_attributes = self._generate_classes_iteratively(classes, domain_context)
            
            # Validate that we got all expected classes
            self._validate_generated_classes(classes, class_attributes)
            
            # Create complete JSON structure with metadata in standard format
            result = self._create_complete_json_structure(class_attributes, domain_context)
            
            # Ensure the result is in standard format by applying conversion if needed
            standard_result = self.convert_to_standard_format(result)
            
            logger.info(f"Successfully generated attributes for {len(classes)} classes in standard format")
            return standard_result
            
        except Exception as e:
            logger.error(f"Failed to generate attributes for multiple classes: {e}")
            raise RuntimeError(f"Failed to generate attributes for multiple classes: {e}") from e
    
    def convert_to_standard_format(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert generated attributes from LLM response format to standard format.
        
        Args:
            attributes: Generated attributes dictionary from LLM
            
        Returns:
            Attributes in standard format
            
        Raises:
            ValueError: If attributes format is invalid
        """
        if not attributes:
            raise ValueError("Attributes dictionary cannot be empty")
        
        try:
            # If already in standard format, return as-is
            if "metadata" in attributes and "classes" in attributes:
                # Check if classes are already properly formatted
                classes = attributes.get("classes", [])
                if classes and all(isinstance(cls, dict) and "name" in cls and "required_attributes" in cls for cls in classes):
                    logger.debug("Attributes already in standard format")
                    return attributes
            
            standard_format = {
                "metadata": attributes.get("metadata", {}),
                "classes": []
            }
            
            for class_response in attributes.get('classes', []):
                try:
                    # Log the structure for debugging
                    logger.debug(f"Processing class response: {type(class_response)}")
                    logger.debug(f"Class response keys: {list(class_response.keys()) if isinstance(class_response, dict) else 'Not a dict'}")
                    
                    # Handle different possible response formats
                    if isinstance(class_response, dict):
                        # Check if it's already in standard format
                        if "name" in class_response and "required_attributes" in class_response:
                            standard_format["classes"].append(class_response)
                            continue
                        
                        # Parse the JSON from the content field if it exists
                        if 'content' in class_response and class_response['content']:
                            content_text = class_response['content'][0]['text']
                            # Extract JSON from markdown code block
                            json_start = content_text.find('```json\n') + 8
                            json_end = content_text.find('\n```')
                            if json_start > 7 and json_end > json_start:
                                json_str = content_text[json_start:json_end]
                                class_data = json.loads(json_str)
                                standard_format["classes"].append(class_data)
                            else:
                                # Try to parse the entire content as JSON
                                class_data = json.loads(content_text)
                                standard_format["classes"].append(class_data)
                        else:
                            # Try to use the response directly
                            standard_format["classes"].append(class_response)
                    else:
                        # If it's a string, try to parse as JSON
                        if isinstance(class_response, str):
                            class_data = json.loads(class_response)
                            standard_format["classes"].append(class_data)
                        
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Error parsing class response: {e}")
                    logger.debug(f"Problematic response: {class_response}")
                    continue
            
            return standard_format
            
        except Exception as e:
            logger.error(f"Failed to convert attributes to standard format: {e}")
            raise ValueError(f"Failed to convert attributes format: {e}") from e

    def save_attributes_to_file(
        self, 
        attributes: Dict[str, Any], 
        file_path: str
    ) -> None:
        """
        Save generated attributes to JSON file.
        
        Args:
            attributes: Generated attributes dictionary
            file_path: Path to save the JSON file
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If file saving fails
        """
        # Validate inputs
        if not attributes:
            raise ValueError("Attributes dictionary cannot be empty")
        if not file_path or not file_path.strip():
            raise ValueError("File path cannot be empty")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to JSON file with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(attributes, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved attributes to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save attributes to {file_path}: {e}")
            raise RuntimeError(f"Failed to save attributes to file: {e}") from e
    

    
    def _generate_classes_iteratively(
        self, 
        classes: List[ClassDefinition], 
        domain_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate attributes for classes sequentially to avoid throttling.
        
        Args:
            classes: List of class definitions
            domain_context: Optional domain context
            
        Returns:
            List of attribute definitions for all classes
        """
        class_attributes = []
        failed_classes = []
        
        # Process classes one by one to avoid throttling
        for i, class_def in enumerate(classes):
            max_retries = config.retry.attribute_generation_max_retries
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    logger.info(f"Processing class {i+1}/{len(classes)}: '{class_def.name}' (attempt {retry_count + 1})")
                    
                    # Generate attributes for single class
                    single_class_attrs = self.generate_attributes_for_class(class_def, domain_context)
                    class_attributes.append(single_class_attrs)
                    
                    logger.info(f"✓ Successfully generated attributes for class '{class_def.name}'")
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"✗ Attempt {retry_count} failed for class '{class_def.name}': {e}")
                    
                    if retry_count < max_retries:
                        # Exponential backoff for retries
                        import time
                        delay = config.retry.attribute_generation_retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"✗ All {max_retries} attempts failed for class '{class_def.name}': {e}")
                        failed_classes.append(class_def.name)
            
            # Add delay between classes to minimize throttling (only if successful)
            if success and i < len(classes) - 1:  # Don't delay after the last class
                import time
                delay = config.retry.attribute_generation_class_delay  # Configurable delay between classes
                logger.info(f"Waiting {delay}s before processing next class...")
                time.sleep(delay)
        
        # Log summary
        logger.info(f"Successfully generated attributes for {len(class_attributes)}/{len(classes)} classes")
        if failed_classes:
            logger.error(f"Failed to generate attributes for {len(failed_classes)} classes: {', '.join(failed_classes)}")
            # Raise an exception if any classes failed to ensure the user is aware
            raise RuntimeError(f"Failed to generate attributes for {len(failed_classes)} classes: {', '.join(failed_classes)}")
        
        return class_attributes
    

    
    def _validate_generated_classes(
        self, 
        input_classes: List[ClassDefinition], 
        generated_attributes: List[Dict[str, Any]]
    ) -> None:
        """
        Validate that all input classes have corresponding generated attributes.
        
        Args:
            input_classes: Original list of class definitions
            generated_attributes: List of generated attribute definitions
            
        Raises:
            RuntimeError: If any classes are missing from the generated attributes
        """
        input_class_names = {cls.name for cls in input_classes}
        generated_class_names = {attrs.get('name', '') for attrs in generated_attributes}
        
        missing_classes = input_class_names - generated_class_names
        
        if missing_classes:
            logger.error(f"Missing classes in generated attributes: {missing_classes}")
            logger.error(f"Expected {len(input_classes)} classes, got {len(generated_attributes)}")
            logger.error(f"Input classes: {sorted(input_class_names)}")
            logger.error(f"Generated classes: {sorted(generated_class_names)}")
            raise RuntimeError(f"Missing {len(missing_classes)} classes in generated attributes: {sorted(missing_classes)}")
        
        logger.info(f"✓ Validation passed: All {len(input_classes)} classes have generated attributes")

    def _create_complete_json_structure(
        self, 
        class_attributes: List[Dict[str, Any]], 
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create complete JSON structure with metadata and all classes.
        
        Args:
            class_attributes: List of class attribute definitions
            domain_context: Optional domain context
            
        Returns:
            Complete JSON structure with metadata
        """
        # Create metadata
        metadata = {
            "domain": domain_context or "document types",
            "num_classes": len(class_attributes),
            "generated_by": "LLM",
            "model": self.model_config.model_id,
            "version": "1.0",
            "generated_at": datetime.now().isoformat()
        }
        
        # Create complete structure
        result = {
            "metadata": metadata,
            "classes": class_attributes
        }
        
        return result
    
    def generate_missing_classes(
        self,
        original_classes: List[ClassDefinition],
        existing_attributes: Dict[str, Any],
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate attributes for classes that are missing from existing attributes file.
        
        This is useful for recovering from partial failures or adding new classes.
        
        Args:
            original_classes: Complete list of class definitions that should exist
            existing_attributes: Existing attributes dictionary (may be incomplete)
            domain_context: Optional domain context
            
        Returns:
            Complete attributes dictionary with all classes
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        if not original_classes:
            raise ValueError("Original classes list cannot be empty")
        if not existing_attributes:
            raise ValueError("Existing attributes cannot be empty")
        
        # Get existing class names
        existing_classes = existing_attributes.get('classes', [])
        existing_class_names = {cls.get('name', '') for cls in existing_classes}
        
        # Find missing classes
        missing_classes = [
            cls for cls in original_classes 
            if cls.name not in existing_class_names
        ]
        
        if not missing_classes:
            logger.info("No missing classes found - all classes already have attributes")
            return existing_attributes
        
        logger.info(f"Found {len(missing_classes)} missing classes: {[cls.name for cls in missing_classes]}")
        
        # Generate attributes for missing classes only
        missing_attributes = self._generate_classes_iteratively(missing_classes, domain_context)
        
        # Combine with existing attributes
        all_classes = existing_classes + missing_attributes
        
        # Update metadata
        updated_metadata = existing_attributes.get('metadata', {}).copy()
        updated_metadata.update({
            "num_classes": len(all_classes),
            "last_updated": datetime.now().isoformat(),
            "recovery_run": True
        })
        
        result = {
            "metadata": updated_metadata,
            "classes": all_classes
        }
        
        # Validate the final result
        self._validate_generated_classes(original_classes, all_classes)
        
        logger.info(f"Successfully recovered {len(missing_classes)} missing classes")
        return result

    def test_bedrock_connection(self) -> bool:
        """
        Test if Bedrock connection is working.
        
        Returns:
            True if connection works, False otherwise
        """
        try:
            self._initialize_bedrock_client()
            
            # Log the actual region and endpoint being used
            logger.info(f"Testing Bedrock connection...")
            logger.info(f"Client region: {self._bedrock_client.meta.region_name}")
            logger.info(f"Client endpoint: {self._bedrock_client.meta.endpoint_url}")
            
            # Make a simple test call
            logger.info("Making test call to Bedrock...")
            
            response = self._bedrock_client.converse(
                modelId=self.model_config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Hello, please respond with just 'OK'"}]
                    }
                ],
                inferenceConfig={
                    "temperature": 0.1,
                    "maxTokens": 10
                }
            )
            
            # Log response metadata to see which region responded
            response_metadata = response.get('ResponseMetadata', {})
            request_id = response_metadata.get('RequestId', 'Unknown')
            
            logger.info("Bedrock test call successful!")
            logger.info(f"Request ID: {request_id}")
            logger.info(f"Response region info: {response_metadata}")
            logger.debug(f"Test response: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Error response: {e.response}")
            return False
    
