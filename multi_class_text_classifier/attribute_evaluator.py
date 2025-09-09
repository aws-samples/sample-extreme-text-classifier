"""
LLM-based attribute evaluator for classification validation.
"""

import json
import logging
import boto3
from typing import Dict, Any, List, Tuple, Union
from botocore.config import Config

from .models.data_models import AttributeValidationResult, AttributeEvaluationConfig
from .config import config


logger = logging.getLogger(__name__)


class LLMAttributeEvaluator:
    """
    LLM-based evaluator for attribute conditions using Amazon Nova Lite via boto3 converse API.
    Evaluates individual conditions with LLM and handles logical operators programmatically.
    Uses lazy evaluation for OR conditions.
    """
    
    def __init__(self, model_config: AttributeEvaluationConfig):
        """
        Initialize LLM evaluator with Amazon Nova Lite configuration.
        
        Args:
            model_config: Configuration for Amazon Nova Lite model
        """
        self.model_config = model_config
        self._bedrock_client = None
        
    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client."""
        if self._bedrock_client is None:
            try:
                # Initialize Bedrock client
                region = config.aws.bedrock_region
                logger.info(f"Initializing Bedrock client in region: {region}")
                
                # Configure retry behavior optimized for Bedrock throttling
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
                
                logger.info(f"Bedrock client initialized successfully with model: {self.model_config.model_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise RuntimeError(f"Failed to initialize Bedrock client: {e}") from e
    
    def evaluate_attributes(
        self, 
        text: str, 
        class_name: str,
        attribute_definition: Dict[str, Any]
    ) -> AttributeValidationResult:
        """
        Evaluate text against attribute definition using LLM for individual conditions.
        
        Args:
            text: Input text to evaluate
            class_name: Name of the class being validated
            attribute_definition: Attribute definition with logical structure
            
        Returns:
            AttributeValidationResult with score and detailed condition results
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM evaluation fails
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if not class_name or not class_name.strip():
            raise ValueError("Class name cannot be empty")
        if not attribute_definition:
            raise ValueError("Attribute definition cannot be empty")
        
        # Initialize Bedrock client if needed
        self._initialize_bedrock_client()
        
        try:
            logger.debug(f"Evaluating attributes for class '{class_name}' with model {self.model_config.model_id}")
            
            # Extract required attributes
            required_attributes = attribute_definition.get("required_attributes", {})
            if not required_attributes:
                raise ValueError("No required_attributes found in attribute definition")
            
            # Evaluate the attribute tree
            evaluation_result = self._evaluate_condition_tree(text, class_name, required_attributes)
            
            # Calculate score based on logical structure
            score = self._calculate_logical_score(evaluation_result["tree"])
            
            # Create result
            result = AttributeValidationResult(
                overall_score=score,
                conditions_met=evaluation_result["conditions_met"],
                conditions_not_met=evaluation_result["conditions_not_met"],
                evaluation_details={
                    "reasoning": evaluation_result["reasoning"],
                    "evaluation_tree": evaluation_result["tree"],
                    "logical_score": score
                }
            )
            
            logger.info(f"Attribute evaluation completed for class '{class_name}' with score {result.overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Attribute evaluation failed for class '{class_name}': {e}")
            raise RuntimeError(f"Failed to evaluate attributes: {e}") from e
    
    def _evaluate_condition_tree(
        self, 
        text: str, 
        class_name: str, 
        condition_node: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Recursively evaluate a condition tree with lazy evaluation.
        
        Args:
            text: Input text to evaluate
            class_name: Name of the class being validated
            condition_node: Either a string condition or a dict with operator and conditions
            
        Returns:
            Dict with evaluation results including satisfied status, conditions met/not met, and reasoning
        """
        if isinstance(condition_node, str):
            # Base case: evaluate individual condition
            return self._evaluate_single_condition(text, class_name, condition_node)
        
        elif isinstance(condition_node, dict) and "operator" in condition_node:
            operator = condition_node["operator"].upper()
            conditions = condition_node.get("conditions", [])
            
            if operator == "AND":
                return self._evaluate_and_conditions(text, class_name, conditions)
            elif operator == "OR":
                return self._evaluate_or_conditions(text, class_name, conditions)
            else:
                raise ValueError(f"Unknown operator: {operator}")
        
        else:
            raise ValueError(f"Invalid condition node format: {condition_node}")
    
    def _evaluate_single_condition(self, text: str, class_name: str, condition: str) -> Dict[str, Any]:
        """
        Evaluate a single condition using LLM.
        
        Args:
            text: Input text to evaluate
            class_name: Name of the class being validated
            condition: Single condition string to evaluate
            
        Returns:
            Dict with evaluation results
        """
        try:
            # Create prompt for single condition evaluation
            from .prompts import AttributeEvaluationPrompts
            prompt = AttributeEvaluationPrompts.single_condition_evaluation_prompt(text, class_name, condition)
            
            # Call Bedrock with Converse API
            response = self._bedrock_client.converse(
                modelId=self.model_config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "temperature": self.model_config.temperature,
                    "maxTokens": self.model_config.max_tokens
                }
            )
            
            # Extract response text from Bedrock Converse API response
            output = response.get('output', {})
            message = output.get('message', {})
            content = message.get('content', [])
            
            response_text = ""
            for content_block in content:
                if 'text' in content_block:
                    response_text = content_block['text']
                    break
            
            # Parse response
            satisfied = self._parse_boolean_response(response_text)

            logger.debug(f"Evaluated single condition. Condition: {condition}, LLM response: {response_text}, Satisfied: {satisfied}")
            
            return {
                "satisfied": satisfied,
                "conditions_met": [condition] if satisfied else [],
                "conditions_not_met": [] if satisfied else [condition],
                "reasoning": f"Condition '{condition}': {'SATISFIED' if satisfied else 'NOT SATISFIED'}",
                "tree": {
                    "type": "condition",
                    "condition": condition,
                    "satisfied": satisfied
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            # Default to not satisfied on error
            return {
                "satisfied": False,
                "conditions_met": [],
                "conditions_not_met": [condition],
                "reasoning": f"Condition '{condition}': ERROR - {str(e)}",
                "tree": {
                    "type": "condition",
                    "condition": condition,
                    "satisfied": False,
                    "error": str(e)
                }
            }
    
    def _evaluate_and_conditions(
        self, 
        text: str, 
        class_name: str, 
        conditions: List[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Evaluate AND conditions (all must be satisfied).
        
        Args:
            text: Input text to evaluate
            class_name: Name of the class being validated
            conditions: List of conditions to evaluate
            
        Returns:
            Dict with evaluation results
        """
        all_conditions_met = []
        all_conditions_not_met = []
        reasoning_parts = []
        tree_results = []
        
        all_satisfied = True
        
        for condition in conditions:
            result = self._evaluate_condition_tree(text, class_name, condition)
            
            all_conditions_met.extend(result["conditions_met"])
            all_conditions_not_met.extend(result["conditions_not_met"])
            reasoning_parts.append(result["reasoning"])
            tree_results.append(result["tree"])
            
            if not result["satisfied"]:
                all_satisfied = False
                # Continue evaluating all conditions for complete feedback
        
        return {
            "satisfied": all_satisfied,
            "conditions_met": all_conditions_met,
            "conditions_not_met": all_conditions_not_met,
            "reasoning": f"({' AND '.join(reasoning_parts)})",
            "tree": {
                "type": "AND",
                "satisfied": all_satisfied,
                "children": tree_results
            }
        }
    
    def _evaluate_or_conditions(
        self, 
        text: str, 
        class_name: str, 
        conditions: List[Union[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Evaluate OR conditions with lazy evaluation (stop at first satisfied condition).
        
        Args:
            text: Input text to evaluate
            class_name: Name of the class being validated
            conditions: List of conditions to evaluate
            
        Returns:
            Dict with evaluation results
        """
        all_conditions_met = []
        all_conditions_not_met = []
        reasoning_parts = []
        tree_results = []
        
        any_satisfied = False
        
        for i, condition in enumerate(conditions):
            result = self._evaluate_condition_tree(text, class_name, condition)
            
            all_conditions_met.extend(result["conditions_met"])
            all_conditions_not_met.extend(result["conditions_not_met"])
            reasoning_parts.append(result["reasoning"])
            tree_results.append(result["tree"])
            
            if result["satisfied"]:
                any_satisfied = True
                # Lazy evaluation: stop at first satisfied condition
                logger.debug(f"OR condition satisfied at index {i}, skipping remaining {len(conditions) - i - 1} conditions")
                
                # Add skipped conditions to tree for completeness
                for j in range(i + 1, len(conditions)):
                    tree_results.append({
                        "type": "condition" if isinstance(conditions[j], str) else "group",
                        "condition": conditions[j] if isinstance(conditions[j], str) else str(conditions[j]),
                        "satisfied": None,
                        "skipped": True
                    })
                break
        
        return {
            "satisfied": any_satisfied,
            "conditions_met": all_conditions_met,
            "conditions_not_met": all_conditions_not_met,
            "reasoning": f"({' OR '.join(reasoning_parts)})",
            "tree": {
                "type": "OR",
                "satisfied": any_satisfied,
                "children": tree_results
            }
        }
    

    
    def _calculate_logical_score(self, tree: Dict[str, Any]) -> float:
        """
        Calculate score based on logical structure of the evaluation tree.
        
        Args:
            tree: Evaluation tree with logical structure
            
        Returns:
            Score between 0.0 and 1.0 based on logical satisfaction
        """
        if tree["type"] == "condition":
            # Single condition: 1.0 if satisfied, 0.0 if not
            return 1.0 if tree["satisfied"] else 0.0
        
        elif tree["type"] == "AND":
            # AND: All children must be satisfied for 1.0, otherwise proportional
            children = tree.get("children", [])
            if not children:
                return 1.0
            
            # Calculate score for each child
            child_scores = [self._calculate_logical_score(child) for child in children]
            
            # For AND: average of all child scores (all must be 1.0 for overall 1.0)
            return sum(child_scores) / len(child_scores)
        
        elif tree["type"] == "OR":
            # OR: If any child is satisfied (score 1.0), the whole OR is satisfied
            children = tree.get("children", [])
            if not children:
                return 1.0
            
            # Calculate score for each child
            child_scores = [self._calculate_logical_score(child) for child in children]
            
            # For OR: maximum of all child scores (any 1.0 makes the whole OR 1.0)
            return max(child_scores)
        
        else:
            # Unknown type - default to satisfied flag if available
            return 1.0 if tree.get("satisfied", False) else 0.0

    def _parse_boolean_response(self, response: str) -> bool:
        """
        Parse LLM response to extract boolean result.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Boolean indicating if condition is satisfied
        """
        # Handle both string and dict responses
        if isinstance(response, dict):
            # If it's a dict, try to extract message or convert to string
            if 'message' in response:
                response_text = response['message']
            else:
                response_text = str(response)
        else:
            response_text = str(response)
        
        # Clean and normalize response
        cleaned_response = response_text.strip().upper()
        
        # Handle various response formats
        if "YES" in cleaned_response:
            return True
        elif "NO" in cleaned_response:
            return False
        else:
            # Default to False if unclear
            logger.warning(f"Unclear LLM response: '{response_text}', defaulting to False")
            return False