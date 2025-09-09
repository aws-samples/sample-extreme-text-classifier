"""
Core data models for the multi-class text classifier.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ClassDefinition:
    """Represents a classification class with its description and metadata."""
    name: str
    description: str
    embedding_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate class definition after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Class name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Class description cannot be empty")


@dataclass
class ClassCandidate:
    """Represents a candidate class with step-specific scores."""
    class_definition: ClassDefinition
    reasoning: str = ""
    attributes_matched: List[str] = field(default_factory=list)
    similarity_score: float = 0.0  # Score from RAG similarity search
    rerank_score: Optional[float] = None  # Score from reranking step
    
    def __post_init__(self):
        """Validate class candidate scores."""
        # Validate step-specific scores
        if not isinstance(self.similarity_score, (int, float)) or not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be a number between 0.0 and 1.0")
        if self.rerank_score is not None and not isinstance(self.rerank_score, (int, float)):
            raise ValueError("Rerank score must be a number or None")
    
    @property
    def effective_score(self) -> float:
        """Get the most relevant score for this candidate."""
        # Use rerank score if available, otherwise similarity score
        return self.rerank_score if self.rerank_score is not None else self.similarity_score


@dataclass
class ProcessingStep:
    """Represents a step in the classification processing pipeline."""
    step_name: str
    input_candidates: List[ClassCandidate]
    output_candidates: List[ClassCandidate]
    processing_time: float
    confidence_change: float


@dataclass
class ClassificationResult:
    """Represents the final classification result with explanations."""
    predicted_class: ClassDefinition
    alternatives: List[ClassCandidate] = field(default_factory=list)
    explanation: str = ""
    processing_steps: List[ProcessingStep] = field(default_factory=list)
    reranked: bool = False
    attribute_validation: Optional['AttributeValidationResult'] = None  # Attribute validation results
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_score(self) -> float:
        """Get the effective score used for this prediction."""
        # Find the predicted class in alternatives to get its effective score
        for alt in self.alternatives:
            if alt.class_definition.name == self.predicted_class.name:
                return alt.effective_score
        # If not found in alternatives, this shouldn't happen, but return 0.0 as fallback
        return 0.0
    
    @property
    def similarity_score(self) -> float:
        """Get the similarity score for this prediction."""
        # Find the predicted class in alternatives to get its similarity score
        for alt in self.alternatives:
            if alt.class_definition.name == self.predicted_class.name:
                return alt.similarity_score
        # If not found in alternatives, this shouldn't happen, but return 0.0 as fallback
        return 0.0
    
    @property
    def rerank_score(self) -> Optional[float]:
        """Get the rerank score for this prediction (None if no reranking was applied)."""
        # Find the predicted class in alternatives to get its rerank score
        for alt in self.alternatives:
            if alt.class_definition.name == self.predicted_class.name:
                return alt.rerank_score
        # If not found in alternatives, return None
        return None
    
    @property
    def attribute_score(self) -> Optional[float]:
        """Get the attribute validation score."""
        return self.attribute_validation.overall_score if self.attribute_validation else None
    
    @classmethod
    def from_candidates(
        cls, 
        candidates: List[ClassCandidate], 
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ClassificationResult':
        """
        Create a ClassificationResult from a list of candidates.
        
        Args:
            candidates: List of class candidates ranked by similarity
            explanation: Optional explanation of the classification
            metadata: Optional metadata dictionary
            
        Returns:
            ClassificationResult with the top candidate as prediction
            
        Raises:
            ValueError: If candidates list is empty
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        # Sort candidates by effective score (rerank score if available, otherwise similarity score)
        sorted_candidates = sorted(candidates, key=lambda c: c.effective_score, reverse=True)
        
        # Top candidate becomes the prediction
        predicted_candidate = sorted_candidates[0]
        predicted_class = predicted_candidate.class_definition
        
        # All candidates become alternatives (including the prediction for easy access to scores)
        alternatives = sorted_candidates
        
        # Generate explanation if not provided
        if not explanation:
            explanation = cls._generate_explanation(predicted_candidate, alternatives)
        
        return cls(
            predicted_class=predicted_class,
            alternatives=alternatives,
            explanation=explanation,
            metadata=metadata or {}
        )
    
    @staticmethod
    def _generate_explanation(
        predicted_candidate: ClassCandidate, 
        alternatives: List[ClassCandidate]
    ) -> str:
        """
        Generate an explanation for the classification result.
        
        Args:
            predicted_candidate: The top candidate
            alternatives: List of alternative candidates
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Main prediction explanation
        explanation_parts.append(
            f"Classified as '{predicted_candidate.class_definition.name}' "
            f"with {predicted_candidate.effective_score:.1%} confidence"
        )
        
        # Add reasoning if available
        if predicted_candidate.reasoning:
            explanation_parts.append(f"Reasoning: {predicted_candidate.reasoning}")
        
        # Add alternatives information
        if alternatives:
            top_alternatives = alternatives[:2]  # Show top 2 alternatives
            alt_names = [f"'{alt.class_definition.name}' ({alt.effective_score:.1%})" 
                        for alt in top_alternatives]
            explanation_parts.append(f"Top alternatives: {', '.join(alt_names)}")
        
        return ". ".join(explanation_parts)
    
    def get_top_candidates(self, include_prediction: bool = True, max_candidates: int = 5) -> List[ClassCandidate]:
        """
        Get the top candidates including the prediction.
        
        Args:
            include_prediction: Whether to include the predicted class as first candidate
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of top candidates sorted by confidence
        """
        candidates = []
        
        # Since alternatives now includes all candidates (including the prediction),
        # we can just use the alternatives directly
        candidates = self.alternatives.copy()
        
        if not include_prediction:
            # Remove the predicted class if not wanted
            candidates = [c for c in candidates if c.class_definition.name != self.predicted_class.name]
        
        # Sort by effective score and limit
        candidates = sorted(candidates, key=lambda c: c.effective_score, reverse=True)
        return candidates[:max_candidates]
    
    def format_result(self, include_alternatives: bool = True, max_alternatives: int = 3) -> str:
        """
        Format the classification result as a human-readable string.
        
        Args:
            include_alternatives: Whether to include alternative candidates
            max_alternatives: Maximum number of alternatives to show
            
        Returns:
            Formatted result string
        """
        lines = []
        
        # Header
        lines.append("🎯 Classification Result")
        lines.append("=" * 50)
        
        # Main prediction
        lines.append(f"📋 Predicted Class: {self.predicted_class.name}")
        lines.append(f"🎯 Effective Score: {self.effective_score:.1%}")
        lines.append(f"📝 Description: {self.predicted_class.description}")
        
        # Attribute validation information
        if self.attribute_validation:
            lines.append(f"✅ Attribute Score: {self.attribute_validation.overall_score:.1%}")
            if self.attribute_validation.conditions_met:
                lines.append(f"   ✓ Conditions Met: {', '.join(self.attribute_validation.conditions_met)}")
            if self.attribute_validation.conditions_not_met:
                lines.append(f"   ✗ Conditions Not Met: {', '.join(self.attribute_validation.conditions_not_met)}")
                # Add explanatory note if conditions are not met but score is 100%
                if self.attribute_validation.overall_score == 1.0:
                    lines.append(f"   ℹ️  Note: Some conditions are not met, but the attribute score is 100%. This can occur when conditions are part of an OR group where only one condition needs to be satisfied.")
        
        # Explanation
        if self.explanation:
            lines.append(f"💡 Explanation: {self.explanation}")
        
        # Alternatives
        if include_alternatives and self.alternatives:
            lines.append(f"\n🔄 Top {min(max_alternatives, len(self.alternatives))} Alternatives:")
            for i, alt in enumerate(self.alternatives[:max_alternatives], 1):
                lines.append(f"   {i}. {alt.class_definition.name} ({alt.effective_score:.1%})")
                lines.append(f"      {alt.class_definition.description[:80]}...")
        
        # Processing steps summary
        if self.processing_steps:
            lines.append(f"\n⚙️ Processing Steps: {len(self.processing_steps)} steps completed")
        
        # Metadata
        if self.metadata:
            lines.append(f"\n📊 Metadata: {len(self.metadata)} items")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the classification result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        result_dict = {
            "predicted_class": {
                "name": self.predicted_class.name,
                "description": self.predicted_class.description,
                "metadata": self.predicted_class.metadata
            },
            "effective_score": self.effective_score,
            "similarity_score": self.similarity_score,
            "rerank_score": self.rerank_score,
            "reranked": self.reranked,
            "explanation": self.explanation,
            "alternatives": [
                {
                    "name": alt.class_definition.name,
                    "description": alt.class_definition.description,
                    "effective_score": alt.effective_score,
                    "similarity_score": alt.similarity_score,
                    "rerank_score": alt.rerank_score,
                    "reasoning": alt.reasoning
                }
                for alt in self.alternatives
            ],
            "processing_steps_count": len(self.processing_steps),
            "metadata": self.metadata
        }
        
        # Add attribute validation data if available
        if self.attribute_validation:
            attribute_validation_dict = {
                "overall_score": self.attribute_validation.overall_score,
                "conditions_met": self.attribute_validation.conditions_met,
                "conditions_not_met": self.attribute_validation.conditions_not_met,
                "evaluation_details": self.attribute_validation.evaluation_details
            }
            
            # Add explanatory note if conditions are not met but score is 100%
            if (self.attribute_validation.conditions_not_met and 
                self.attribute_validation.overall_score == 1.0):
                attribute_validation_dict["note"] = (
                    "Some conditions are not met, but the attribute score is 100%. "
                    "This can occur when conditions are part of an OR group where only one condition needs to be satisfied."
                )
            
            result_dict["attribute_validation"] = attribute_validation_dict
            result_dict["attribute_score"] = self.attribute_validation.overall_score
        else:
            result_dict["attribute_score"] = None
        
        return result_dict
    
    def get_confidence_level(self) -> str:
        """
        Get a human-readable confidence level description.
        
        Returns:
            Confidence level as string (High, Medium, Low, Very Low)
        """
        if self.effective_score >= 0.8:
            return "High"
        elif self.effective_score >= 0.6:
            return "Medium"
        elif self.effective_score >= 0.4:
            return "Low"
        else:
            return "Very Low"


@dataclass
class RerankingConfig:
    """Configuration for LLM-based reranking of classification candidates."""
    model_type: str  # "llm" for generic LLMs via Strands, "amazon_rerank", "cohere_rerank"
    top_k_candidates: int = 5  # Number of candidates to rerank
    model_id: Optional[str] = None  # Model ID for Strands LLMs (e.g., "us.amazon.nova-lite-v1:0")
    aws_region: Optional[str] = None  # For Bedrock models via Strands
    api_key: Optional[str] = None  # For external APIs (Amazon Rerank, Cohere)
    model_parameters: Optional[Dict[str, Any]] = None  # Model-specific parameters
    fallback_on_error: bool = True  # Whether to fall back to similarity search on error
    
    def __post_init__(self):
        """Validate reranking configuration after initialization."""
        valid_model_types = {"llm", "amazon_rerank", "cohere_rerank"}
        if self.model_type not in valid_model_types:
            raise ValueError(f"Model type must be one of {valid_model_types}, got '{self.model_type}'")
        
        if self.top_k_candidates <= 0:
            raise ValueError("top_k_candidates must be positive")
        
        if self.model_type == "llm" and not self.model_id:
            raise ValueError("model_id is required for llm model type")
        
        # Both Amazon Rerank and Cohere Rerank use boto3 with AWS credentials via Bedrock
        # No API key validation needed for bedrock-based reranking services


@dataclass
class ClassifierConfig:
    """Configuration options for the text classifier."""
    # Architecture options
    enable_reranking: bool = False
    enable_attribute_matching: bool = False
    
    # RAG configuration
    rag_retrieval_k: int = 50
    
    # Reranking configuration
    reranking_model: Optional[str] = None
    reranking_with_llm: bool = False
    reranking_top_k: int = 10
    
    # S3 configuration for vectors and attributes
    s3_bucket: str = ""
    s3_vectors_prefix: str = "vectors/"
    s3_attributes_prefix: str = "attributes/"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.rag_retrieval_k <= 0:
            raise ValueError("RAG retrieval k must be positive")
        if self.reranking_top_k <= 0:
            raise ValueError("Reranking top k must be positive")


@dataclass
class TestSample:
    """Represents a test sample for evaluation."""
    text: str
    ground_truth_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Represents a benchmarking report with performance metrics."""
    configurations_tested: List[str]
    accuracy_metrics: Dict[str, float]
    performance_stats: Dict[str, Any]
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AttributeValidationResult:
    """Result of attribute validation for a single class."""
    overall_score: float  # 0.0 to 1.0
    conditions_met: List[str]  # List of conditions that were satisfied
    conditions_not_met: List[str]  # List of conditions that were not satisfied
    evaluation_details: Dict[str, Any]  # Additional details from LLM evaluation
    
    def __post_init__(self):
        """Validate attribute validation result after initialization."""
        if not isinstance(self.overall_score, (int, float)) or not (0.0 <= self.overall_score <= 1.0):
            raise ValueError("Overall score must be a number between 0.0 and 1.0")
        if not isinstance(self.conditions_met, list):
            raise ValueError("conditions_met must be a list")
        if not isinstance(self.conditions_not_met, list):
            raise ValueError("conditions_not_met must be a list")
        if not isinstance(self.evaluation_details, dict):
            raise ValueError("evaluation_details must be a dictionary")


@dataclass
class AttributeEvaluationConfig:
    """Configuration for LLM-based attribute evaluation."""
    model_id: str = "us.amazon.nova-lite-v1:0"  # Amazon Nova Lite
    temperature: float = 0.1
    max_tokens: int = 1000
    
    def __post_init__(self):
        """Validate attribute evaluation configuration after initialization."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        if not isinstance(self.temperature, (int, float)) or not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be a number between 0.0 and 2.0")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")


@dataclass
class AttributeGenerationConfig:
    """Configuration for automatic attribute generation."""
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # Claude Sonnet 4
    temperature: float = 0.1
    max_tokens: int = 4000
    
    def __post_init__(self):
        """Validate attribute generation configuration after initialization."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        if not isinstance(self.temperature, (int, float)) or not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be a number between 0.0 and 2.0")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")


@dataclass
class AttributeValidationConfig:
    """Configuration for attribute validation feature."""
    enabled: bool = False
    attributes_path: str = ""
    evaluation_config: Optional[AttributeEvaluationConfig] = None
    
    def __post_init__(self):
        """Validate attribute validation configuration after initialization."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.enabled and (not self.attributes_path or not self.attributes_path.strip()):
            raise ValueError("attributes_path is required when attribute validation is enabled")
        if self.evaluation_config is not None and not isinstance(self.evaluation_config, AttributeEvaluationConfig):
            raise ValueError("evaluation_config must be an AttributeEvaluationConfig instance or None")