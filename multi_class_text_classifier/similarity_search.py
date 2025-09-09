"""
Similarity search functionality for multi-class text classification.

This module provides cosine similarity calculations and ranking of classes
based on embedding vectors.
"""

import math
import numpy as np
from typing import List, Optional
from .models.data_models import ClassDefinition, ClassCandidate
from .exceptions import ClassifierError


class SimilaritySearchError(ClassifierError):
    """Raised when similarity search operations fail."""
    pass


class SimilaritySearch:
    """
    Handles similarity search operations using cosine similarity.
    
    This class provides methods to find the most similar classes to a given
    text input based on embedding vectors using cosine similarity calculations.
    """
    
    def __init__(self, classes: List[ClassDefinition]):
        """
        Initialize similarity search with class embeddings.
        
        Args:
            classes: List of classes with embeddings
            
        Raises:
            SimilaritySearchError: If classes list is empty or contains invalid data
        """
        if not classes:
            raise SimilaritySearchError("Classes list cannot be empty")
        
        # Validate that all classes have embeddings
        classes_without_embeddings = [c for c in classes if c.embedding_vector is None]
        if classes_without_embeddings:
            class_names = [c.name for c in classes_without_embeddings[:3]]
            raise SimilaritySearchError(
                f"Classes without embeddings found: {class_names}"
                f"{'...' if len(classes_without_embeddings) > 3 else ''}"
            )
        
        self.classes = classes
        self._validate_embedding_dimensions()
        
        # Pre-compute embedding matrix for efficient similarity calculations
        self._embedding_matrix = self._build_embedding_matrix()
        self._embedding_norms = self._compute_embedding_norms()
    
    def _validate_embedding_dimensions(self) -> None:
        """
        Validate that all embeddings have the same dimensions.
        
        Raises:
            SimilaritySearchError: If embeddings have inconsistent dimensions
        """
        if not self.classes:
            return
        
        expected_dim = len(self.classes[0].embedding_vector)
        for class_def in self.classes:
            if len(class_def.embedding_vector) != expected_dim:
                raise SimilaritySearchError(
                    f"Inconsistent embedding dimensions: expected {expected_dim}, "
                    f"got {len(class_def.embedding_vector)} for class '{class_def.name}'"
                )
    
    def _build_embedding_matrix(self) -> np.ndarray:
        """
        Build a numpy matrix of all class embeddings for efficient computation.
        
        Returns:
            Numpy array of shape (num_classes, embedding_dim)
        """
        try:
            embedding_matrix = np.array([class_def.embedding_vector for class_def in self.classes])
            return embedding_matrix
        except Exception as e:
            raise SimilaritySearchError(f"Error building embedding matrix: {e}")
    
    def _compute_embedding_norms(self) -> np.ndarray:
        """
        Pre-compute L2 norms of all class embeddings for efficient cosine similarity.
        
        Returns:
            Numpy array of L2 norms for each class embedding
        """
        try:
            norms = np.linalg.norm(self._embedding_matrix, axis=1)
            # Handle zero norms to avoid division by zero
            from .config import config
            norms = np.where(norms == 0, config.similarity.zero_norm_epsilon, norms)
            return norms
        except Exception as e:
            raise SimilaritySearchError(f"Error computing embedding norms: {e}")
    
    def find_similar_classes(self, text_embedding: List[float], top_k: int = None) -> List[ClassCandidate]:
        """
        Find most similar classes using cosine similarity with optimized numpy operations.
        
        Args:
            text_embedding: Embedding vector for the input text
            top_k: Number of top candidates to return (defaults to config value)
            
        Returns:
            List of ClassCandidate objects sorted by similarity score (highest first)
            
        Raises:
            SimilaritySearchError: If search fails or inputs are invalid
        """
        from .config import config
        
        if top_k is None:
            top_k = config.similarity.default_top_k
        """
        Find most similar classes using cosine similarity with optimized numpy operations.
        
        Args:
            text_embedding: Embedding vector for input text
            top_k: Number of top candidates to return
            
        Returns:
            List of class candidates ranked by similarity (highest first)
            
        Raises:
            SimilaritySearchError: If text_embedding is invalid or top_k is invalid
        """
        if not text_embedding:
            raise SimilaritySearchError("Text embedding cannot be empty")
        
        if top_k <= 0:
            raise SimilaritySearchError("top_k must be positive")
        
        if top_k > len(self.classes):
            top_k = len(self.classes)
        
        try:
            # Convert text embedding to numpy array
            text_embedding_np = np.array(text_embedding)
            
            # Validate dimensions match
            if text_embedding_np.shape[0] != self._embedding_matrix.shape[1]:
                raise SimilaritySearchError(
                    f"Text embedding dimension {text_embedding_np.shape[0]} doesn't match "
                    f"class embedding dimension {self._embedding_matrix.shape[1]}"
                )
            
            # Calculate text embedding norm
            text_norm = np.linalg.norm(text_embedding_np)
            if text_norm == 0:
                text_norm = config.similarity.zero_norm_epsilon  # Handle zero norm
            
            # Vectorized cosine similarity calculation using broadcasting
            # dot_products shape: (num_classes,)
            dot_products = np.dot(self._embedding_matrix, text_embedding_np)
            
            # Cosine similarities shape: (num_classes,)
            cosine_similarities = dot_products / (self._embedding_norms * text_norm)
            
            # Normalize to 0-1 range (from -1 to 1)
            normalized_similarities = (cosine_similarities + 1.0) / 2.0
            
            # Clamp to [0, 1] range to handle floating point precision issues
            normalized_similarities = np.clip(normalized_similarities, 0.0, 1.0)
            
            # Get top k indices using argpartition for efficiency
            if top_k < len(self.classes):
                # Use argpartition for better performance when top_k << num_classes
                top_k_indices = np.argpartition(normalized_similarities, -top_k)[-top_k:]
                # Sort the top k indices by similarity score (descending)
                top_k_indices = top_k_indices[np.argsort(normalized_similarities[top_k_indices])[::-1]]
            else:
                # If we need all classes, just sort all indices
                top_k_indices = np.argsort(normalized_similarities)[::-1]
            
            # Create ClassCandidate objects for top k results
            candidates = []
            for idx in top_k_indices:
                similarity_score = float(normalized_similarities[idx])
                
                candidate = ClassCandidate(
                    class_definition=self.classes[idx],
                    similarity_score=similarity_score,
                    reasoning=f"Cosine similarity: {similarity_score:.4f}"
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            raise SimilaritySearchError(f"Error finding similar classes: {e}")
    
    def get_class_count(self) -> int:
        """
        Get the number of classes available for similarity search.
        
        Returns:
            Number of classes with embeddings
        """
        return len(self.classes)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension, or None if no classes available
        """
        if not self.classes:
            return None
        return len(self.classes[0].embedding_vector)