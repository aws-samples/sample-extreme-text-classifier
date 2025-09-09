"""
Prompt templates for the multi-class text classifier backend.
Contains all LLM prompts used throughout the system.
"""

from typing import Dict, Any, List, Optional





class ExampleGenerationPrompts:
    """Prompts for example generation."""
    
    @staticmethod
    def batch_example_generation_prompt(
        classes: List[Dict[str, str]],
        domain: str,
        num_examples_per_class: int
    ) -> str:
        """Generate prompt for batch example generation."""
        class_descriptions = []
        for i, cls in enumerate(classes, 1):
            class_descriptions.append(f"{i}. {cls['name']}: {cls['description']}")
        
        classes_text = "\n".join(class_descriptions)
        
        return f"""Generate {num_examples_per_class} realistic example document(s) for each of the following classification classes in the {domain} domain:

Classes:
{classes_text}

Requirements:
- Each example should be a realistic text that would belong to its specific class
- Each example will be used to benchmark a classification system, hence its class should not be too obvious
- Keep each example between 50-200 words
- Make examples specific to the {domain} domain
- Examples should be diverse and representative of each class
- Use format appropriate for the domain (product specs, technical document extract...)
- Include relevant details, terminology, and context for the domain
- Ensure examples clearly differentiate between classes

Use the generate_examples tool to provide structured output with examples organized by class name."""
    
    @staticmethod
    def single_class_example_generation_prompt(
        class_name: str,
        class_description: str,
        domain: str,
        examples_per_class: int
    ) -> str:
        """Generate prompt for single class example generation."""
        return f"""Generate {examples_per_class} realistic example documents for the following class in the {domain} domain:

Class Name: {class_name}
Class Description: {class_description}

Requirements:
- Each example should be a realistic document that would belong to this class
- Examples should be diverse and representative of the class
- Keep each example between 50-200 words
- Make examples specific to the {domain} domain
- Return only the example text, one per line, numbered 1-{examples_per_class}

Examples:"""
    
    @staticmethod
    def connection_test_prompt() -> str:
        """Simple prompt for testing connection."""
        return "Hello, please respond with 'Connection successful'"


class ClassificationPrompts:
    """Prompts for classification tasks."""
    
    @staticmethod
    def reranking_prompt(
        text: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> str:
        """Generate prompt for LLM-based reranking."""
        candidates_text = []
        for i, candidate in enumerate(candidates, 1):
            candidates_text.append(
                f"{i}. {candidate['name']}: {candidate['description']}"
            )
        
        candidates_list = "\n".join(candidates_text)
        
        return f"""You are an expert document classifier. Rerank the following candidate classes based on how well they match the given text.

Text to classify:
{text}

Candidate classes:
{candidates_list}

Instructions:
1. Analyze the text content, structure, and purpose
2. Consider which class best describes the document type
3. Rank the top {top_k} most relevant classes
4. Provide reasoning for your rankings

Return your response using the rerank_candidates tool with the ranked list."""


class ValidationPrompts:
    """Prompts for validation tasks."""
    
    @staticmethod
    def attribute_validation_prompt(
        text: str,
        class_name: str,
        attributes: Dict[str, Any]
    ) -> str:
        """Generate prompt for attribute validation."""
        return f"""You are an expert document classifier. Evaluate whether the given text meets the attribute requirements for the class "{class_name}".

Text to evaluate:
{text}

Class: {class_name}
Required Attributes: {attributes}

Instructions:
1. Carefully read the text and understand its content
2. Check each attribute condition against the text
3. For AND operators, all conditions must be met
4. For OR operators, at least one condition must be met
5. Provide a score from 0.0 to 1.0 indicating how well the text matches the attributes
6. List which conditions are met and which are not met

Use the validate_attributes tool to provide your structured response."""


class SystemPrompts:
    """System-level prompts and messages."""
    
    @staticmethod
    def fallback_example_template(class_name: str, domain: str, example_number: int) -> str:
        """Template for fallback examples when generation fails."""
        return f"Example document for {class_name} in the {domain} domain (example {example_number})"
    
    @staticmethod
    def professional_fallback_template(class_name: str, domain: str) -> str:
        """Template for professional fallback examples."""
        return f"Professional document example for {class_name} in the {domain} domain"


# Tool definitions for structured responses
class ToolDefinitions:
    """Tool definitions for structured LLM responses."""
    

    
    @staticmethod
    def generate_examples_tool() -> Dict[str, Any]:
        """Tool definition for example generation."""
        return {
            "toolSpec": {
                "name": "generate_examples",
                "description": "Generate examples for multiple classification classes",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "examples": {
                                "type": "object",
                                "description": "Examples organized by class name",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A realistic example document for this class"
                                    }
                                }
                            }
                        },
                        "required": ["examples"]
                    }
                }
            }
        }
    
    @staticmethod
    def rerank_candidates_tool() -> Dict[str, Any]:
        """Tool definition for candidate reranking."""
        return {
            "toolSpec": {
                "name": "rerank_candidates",
                "description": "Rerank classification candidates by relevance",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ranked_candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "class_name": {"type": "string"},
                                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                                        "reasoning": {"type": "string"}
                                    },
                                    "required": ["class_name", "score", "reasoning"]
                                }
                            }
                        },
                        "required": ["ranked_candidates"]
                    }
                }
            }
        }
    
    @staticmethod
    def validate_attributes_tool() -> Dict[str, Any]:
        """Tool definition for attribute validation."""
        return {
            "toolSpec": {
                "name": "validate_attributes",
                "description": "Validate text against class attributes",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "overall_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Overall attribute match score"
                            },
                            "conditions_met": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of conditions that were satisfied"
                            },
                            "conditions_not_met": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of conditions that were not satisfied"
                            },
                            "evaluation_details": {
                                "type": "object",
                                "description": "Additional evaluation details"
                            }
                        },
                        "required": ["overall_score", "conditions_met", "conditions_not_met", "evaluation_details"]
                    }
                }
            }
        }