"""
Core interfaces for the multi-class text classifier services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models import ClassDefinition, ClassCandidate, ClassifierConfig


class RAGPipelineInterface(ABC):
    """Interface for RAG (Retrieval-Augmented Generation) pipeline."""
    
    @abstractmethod
    def process(self, text: str) -> List[ClassCandidate]:
        """
        Process text through RAG pipeline to generate class candidates.
        
        Args:
            text: Input text to classify
            
        Returns:
            List of class candidates with confidence scores
        """
        pass
    
    @abstractmethod
    def generate_class_embeddings(self) -> None:
        """Generate and store embeddings for all class descriptions."""
        pass


class RerankingModuleInterface(ABC):
    """Interface for reranking classification candidates."""
    
    @abstractmethod
    def rerank(self, 
               text: str, 
               candidates: List[ClassCandidate],
               attributes: Optional[Dict] = None) -> List[ClassCandidate]:
        """
        Rerank classification candidates using configured models and attributes.
        
        Args:
            text: Original input text
            candidates: List of class candidates to rerank
            attributes: Optional attributes for matching
            
        Returns:
            Reranked list of class candidates
        """
        pass


class AttributeExtractorInterface(ABC):
    """Interface for extracting and matching attributes."""
    
    @abstractmethod
    def extract_class_attributes(self, classes: List[ClassDefinition]) -> Dict[str, Dict]:
        """
        Extract semantic attributes from class descriptions.
        
        Args:
            classes: List of class definitions
            
        Returns:
            Dictionary mapping class names to their attributes
        """
        pass
    
    @abstractmethod
    def match_attributes(self, 
                        text: str, 
                        candidates: List[ClassCandidate]) -> List[ClassCandidate]:
        """
        Match text attributes against class attributes to refine rankings.
        
        Args:
            text: Input text to analyze
            candidates: List of class candidates
            
        Returns:
            Updated candidates with attribute matching scores
        """
        pass


class ConfidenceEvaluatorInterface(ABC):
    """Interface for evaluating classification confidence."""
    
    @abstractmethod
    def evaluate_confidence(self, candidates: List[ClassCandidate]) -> float:
        """
        Calculate overall confidence score for classification.
        
        Args:
            candidates: List of class candidates
            
        Returns:
            Overall confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def is_uncertain(self, confidence: float, threshold: float) -> bool:
        """
        Determine if classification is uncertain based on confidence threshold.
        
        Args:
            confidence: Confidence score
            threshold: Uncertainty threshold
            
        Returns:
            True if classification is uncertain
        """
        pass


class VectorStoreInterface(ABC):
    """Interface for vector storage operations."""
    
    @abstractmethod
    def save_vectors(self, vectors: Dict[str, List[float]]) -> None:
        """
        Save vectors to storage.
        
        Args:
            vectors: Dictionary mapping identifiers to vectors
        """
        pass
    
    @abstractmethod
    def load_vectors(self) -> Dict[str, List[float]]:
        """
        Load vectors from storage.
        
        Returns:
            Dictionary mapping identifiers to vectors
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query_vector: List[float], k: int) -> List[tuple]:
        """
        Perform similarity search for k nearest vectors.
        
        Args:
            query_vector: Query vector for similarity search
            k: Number of nearest neighbors to return
            
        Returns:
            List of (identifier, similarity_score) tuples
        """
        pass