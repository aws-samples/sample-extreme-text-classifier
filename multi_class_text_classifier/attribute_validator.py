"""
Attribute validator for classification results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .models.data_models import (
    AttributeValidationResult, 
    AttributeEvaluationConfig, 
    ClassDefinition
)
from .attribute_evaluator import LLMAttributeEvaluator


logger = logging.getLogger(__name__)


class AttributeValidator:
    """
    Validates classification predictions against attribute definitions.
    
    This class loads attribute definitions from a JSON file and uses an LLM-based
    evaluator to check if classified text matches the required attributes for the
    predicted class.
    """
    
    def __init__(self, attributes_path: str, model_config: AttributeEvaluationConfig):
        """
        Initialize the attribute validator.
        
        Args:
            attributes_path: Path to JSON file containing attribute definitions
            model_config: Configuration for LLM-based attribute evaluation
            
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If attributes file doesn't exist
            RuntimeError: If attribute definitions cannot be loaded
        """
        # Validate inputs
        if not attributes_path or not attributes_path.strip():
            raise ValueError("Attributes path cannot be empty")
        if not isinstance(model_config, AttributeEvaluationConfig):
            raise ValueError("model_config must be an AttributeEvaluationConfig instance")
        
        self.attributes_path = attributes_path
        self.model_config = model_config
        self.attribute_definitions = {}
        
        # Initialize LLM evaluator
        self.llm_evaluator = LLMAttributeEvaluator(model_config)
        
        # Load attribute definitions
        self._load_attribute_definitions()
        
        logger.info(f"AttributeValidator initialized with {len(self.attribute_definitions)} class definitions")
    
    def _load_attribute_definitions(self):
        """
        Load attribute definitions from JSON file.
        
        Raises:
            FileNotFoundError: If attributes file doesn't exist
            RuntimeError: If JSON cannot be parsed or has invalid structure
        """
        try:
            # Check if file exists
            attributes_file = Path(self.attributes_path)
            if not attributes_file.exists():
                raise FileNotFoundError(f"Attributes file not found: {self.attributes_path}")
            
            # Load and parse JSON
            with open(attributes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(data, dict):
                raise RuntimeError("Attribute definitions must be a JSON object")
            
            if "classes" not in data:
                raise RuntimeError("Attribute definitions must contain 'classes' field")
            
            if not isinstance(data["classes"], list):
                raise RuntimeError("'classes' field must be a list")
            
            # Build lookup dictionary by class name
            self.attribute_definitions = {}
            for class_def in data["classes"]:
                if not isinstance(class_def, dict):
                    logger.warning(f"Skipping invalid class definition: {class_def}")
                    continue
                
                class_name = class_def.get("name")
                if not class_name:
                    logger.warning(f"Skipping class definition without name: {class_def}")
                    continue
                
                # Validate required_attributes structure
                required_attributes = class_def.get("required_attributes")
                if not required_attributes:
                    logger.warning(f"Class '{class_name}' has no required_attributes, skipping")
                    continue
                
                self.attribute_definitions[class_name] = class_def
                logger.debug(f"Loaded attribute definition for class: {class_name}")
            
            if not self.attribute_definitions:
                raise RuntimeError("No valid class definitions found in attributes file")
            
            logger.info(f"Successfully loaded {len(self.attribute_definitions)} attribute definitions")
            
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON from attributes file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load attribute definitions: {e}") from e
    
    def validate_prediction(
        self, 
        text: str, 
        predicted_class: ClassDefinition
    ) -> AttributeValidationResult:
        """
        Validate the predicted class against its attribute definition.
        
        Args:
            text: Original input text being classified
            predicted_class: The predicted class to validate
            
        Returns:
            AttributeValidationResult with score and condition details
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If validation fails due to system errors
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if not isinstance(predicted_class, ClassDefinition):
            raise ValueError("predicted_class must be a ClassDefinition instance")
        if not predicted_class.name or not predicted_class.name.strip():
            raise ValueError("Predicted class name cannot be empty")
        
        try:
            logger.debug(f"Validating prediction for class: {predicted_class.name}")
            
            # Find attribute definition for the predicted class
            class_name = predicted_class.name
            attribute_definition = self.attribute_definitions.get(class_name)
            
            if not attribute_definition:
                logger.warning(f"No attribute definition found for class '{class_name}'")
                # Return a result indicating no validation was performed
                from .config import config
                return AttributeValidationResult(
                    overall_score=config.validation.default_pass_score,  # Default to passing when no attributes defined
                    conditions_met=[],
                    conditions_not_met=[],
                    evaluation_details={
                        "reasoning": f"No attribute definition found for class '{class_name}'",
                        "validation_performed": False
                    }
                )
            
            # Use LLM evaluator to validate attributes
            logger.debug(f"Evaluating attributes for class '{class_name}' using LLM")
            result = self.llm_evaluator.evaluate_attributes(
                text=text,
                class_name=class_name,
                attribute_definition=attribute_definition
            )
            
            # Add validation metadata
            result.evaluation_details["validation_performed"] = True
            result.evaluation_details["class_name"] = class_name
            result.evaluation_details["attributes_file"] = self.attributes_path
            
            logger.info(
                f"Attribute validation completed for class '{class_name}': "
                f"score={result.overall_score:.2f}, "
                f"conditions_met={len(result.conditions_met)}, "
                f"conditions_not_met={len(result.conditions_not_met)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Attribute validation failed for class '{predicted_class.name}': {e}")
            # Return a failed result with error information
            return AttributeValidationResult(
                overall_score=config.validation.default_fail_score,
                conditions_met=[],
                conditions_not_met=[],
                evaluation_details={
                    "reasoning": f"Validation failed due to error: {str(e)}",
                    "validation_performed": False,
                    "error": str(e)
                }
            )
    
    def get_available_classes(self) -> list[str]:
        """
        Get list of class names that have attribute definitions.
        
        Returns:
            List of class names with attribute definitions
        """
        return list(self.attribute_definitions.keys())
    
    def has_attributes_for_class(self, class_name: str) -> bool:
        """
        Check if attribute definitions exist for a given class.
        
        Args:
            class_name: Name of the class to check
            
        Returns:
            True if attribute definitions exist for the class, False otherwise
        """
        return class_name in self.attribute_definitions
    
    def get_class_attributes(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get attribute definition for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Attribute definition dictionary or None if not found
        """
        return self.attribute_definitions.get(class_name)