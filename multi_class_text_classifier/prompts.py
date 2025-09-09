"""
Prompt templates for the multi-class text classifier library.
Contains all LLM prompts used throughout the system.
"""

from typing import Dict, Any, Optional, List


class AttributeGenerationPrompts:
    """Prompts for attribute generation."""
    
    @staticmethod
    def single_class_generation_prompt(
        class_name: str,
        class_description: str,
        domain_context: Optional[str] = None
    ) -> str:
        """Generate prompt for single class attribute generation."""
        domain_info = f"\nDomain Context: {domain_context}" if domain_context else ""
        
        return f"""You are an expert in document classification. Generate logical attribute requirements for this document class.

Class: {class_name}
Description: {class_description}{domain_info}

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


class AttributeEvaluationPrompts:
    """Prompts for attribute evaluation."""
    
    @staticmethod
    def system_prompt() -> str:
        """System prompt for attribute evaluation agent."""
        return "You are an expert document classifier. Evaluate whether text satisfies specific conditions."
    
    @staticmethod
    def single_condition_evaluation_prompt(
        text: str,
        class_name: str,
        condition: str
    ) -> str:
        """Create prompt for evaluating a single condition."""
        return f"""You are evaluating whether a document satisfies a specific condition for the class "{class_name}".

TEXT TO EVALUATE:
{text}

CONDITION TO CHECK:
{condition}

INSTRUCTIONS:
1. Carefully read the text and understand its content
2. Determine if the text satisfies the given condition
3. Respond with ONLY "YES" if the condition is satisfied, or "NO" if it is not satisfied
4. Do not provide any explanation or additional text

Your response (YES or NO):"""


class RerankingPrompts:
    """Prompts for LLM-based reranking."""
    
    @staticmethod
    def system_prompt() -> str:
        """System prompt for reranking agent."""
        return """You are a text classification expert. Your task is to rerank classification candidates based on how well they match the input text.

You will be given:
1. An input text to classify
2. A list of candidate classes with their descriptions

Your job is to analyze the semantic meaning of the input text and rank the candidates from most relevant to least relevant. Consider:
- Semantic similarity between input text and class descriptions
- Context and domain relevance
- Specific keywords and concepts that match

Respond with a JSON object containing the reranked candidates with scores between 0.0 and 1.0, where 1.0 is the most relevant."""
    
    @staticmethod
    def reranking_prompt(
        text: str,
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for LLM-based reranking."""
        candidates_text = []
        for i, candidate in enumerate(candidates, 1):
            candidates_text.append(
                f"{i}. {candidate['name']}: {candidate['description']}"
            )
        
        candidates_list = "\n".join(candidates_text)
        
        return f"""Text to classify:
{text}

Candidate classes:
{candidates_list}

Please analyze the text and rank these candidates by relevance. Respond with a JSON object containing:
{{
  "rankings": [
    {{
      "class_name": "exact class name",
      "score": 0.95,
      "reasoning": "explanation for this ranking"
    }}
  ]
}}

Requirements:
- Include ALL candidates in your rankings
- Scores should be between 0.0 and 1.0 (1.0 = most relevant)
- Order by relevance (highest score first)
- Provide clear reasoning for each ranking"""


class DatasetGenerationPrompts:
    """Prompts for dataset generation."""
    
    @staticmethod
    def system_prompt() -> str:
        """System prompt for dataset generation agent."""
        return "You are a helpful assistant that generates diverse classification datasets for given domains. Always respond with valid JSON format."
    
    @staticmethod
    def dataset_generation_prompt(
        domain: str,
        num_classes: int
    ) -> str:
        """Generate prompt for dataset generation."""
        return f"""Generate {num_classes} diverse and distinct classification classes for the domain: {domain}

Requirements:
1. Each class should have a clear, specific name (2-4 words maximum)
2. Each class should have a detailed description (20-50 words) explaining what content belongs to this class
3. Classes should be diverse and cover different aspects of the {domain} domain
4. Classes should be mutually exclusive (no overlap)
5. Classes should be close to eachother to make it challenging to classify, as this dataset will be used for benchmarking
6. Descriptions should be specific enough to distinguish between classes
7. Use professional, clear language appropriate for the domain

Output format (JSON):
{{
  "classes": [
    {{
      "name": "Class Name",
      "description": "Detailed description of what content belongs to this classification class"
    }}
  ]
}}

Generate exactly {num_classes} classes covering the breadth of the {domain} domain. Make sure each class is unique and well-defined."""


class ToolDefinitions:
    """Tool definitions for structured LLM responses."""
    
    @staticmethod
    def generate_class_attributes_tool() -> Dict[str, Any]:
        """Tool definition for class attribute generation."""
        return {
            "toolSpec": {
                "name": "generate_class_attributes",
                "description": "Generate attribute definition for a document class",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The class name"
                            },
                            "description": {
                                "type": "string",
                                "description": "The class description"
                            },
                            "required_attributes": {
                                "type": "object",
                                "properties": {
                                    "operator": {
                                        "type": "string",
                                        "enum": ["AND", "OR"],
                                        "description": "Logical operator for combining conditions"
                                    },
                                    "conditions": {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                {
                                                    "type": "string",
                                                    "description": "A single condition string"
                                                },
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "operator": {
                                                            "type": "string",
                                                            "enum": ["AND", "OR"]
                                                        },
                                                        "conditions": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        }
                                                    },
                                                    "required": ["operator", "conditions"]
                                                }
                                            ]
                                        },
                                        "description": "List of conditions or nested logical structures"
                                    }
                                },
                                "required": ["operator", "conditions"]
                            }
                        },
                        "required": ["name", "description", "required_attributes"]
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
                            "rankings": {
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
                        "required": ["rankings"]
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