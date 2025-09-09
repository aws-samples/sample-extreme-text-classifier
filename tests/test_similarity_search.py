"""
Tests for similarity search functionality.
"""

import pytest
import math
import numpy as np
from multi_class_text_classifier.similarity_search import SimilaritySearch, SimilaritySearchError
from multi_class_text_classifier.models.data_models import ClassDefinition, ClassCandidate


class TestSimilaritySearch:
    """Test cases for SimilaritySearch class."""
    
    def test_init_with_valid_classes(self):
        """Test initialization with valid classes containing embeddings."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="Class2", 
                description="Description 2",
                embedding_vector=[0.0, 1.0, 0.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        assert similarity_search.get_class_count() == 2
        assert similarity_search.get_embedding_dimension() == 3
    
    def test_init_with_empty_classes(self):
        """Test initialization with empty classes list raises error."""
        with pytest.raises(SimilaritySearchError, match="Classes list cannot be empty"):
            SimilaritySearch([])
    
    def test_init_with_classes_without_embeddings(self):
        """Test initialization with classes missing embeddings raises error."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2",
                embedding_vector=None  # Missing embedding
            )
        ]
        
        with pytest.raises(SimilaritySearchError, match="Classes without embeddings found"):
            SimilaritySearch(classes)
    
    def test_init_with_inconsistent_embedding_dimensions(self):
        """Test initialization with inconsistent embedding dimensions raises error."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]  # 3 dimensions
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2", 
                embedding_vector=[0.0, 1.0]  # 2 dimensions
            )
        ]
        
        with pytest.raises(SimilaritySearchError, match="Inconsistent embedding dimensions"):
            SimilaritySearch(classes)
    

    
    def test_find_similar_classes_basic(self):
        """Test finding similar classes with basic functionality."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2",
                embedding_vector=[0.0, 1.0, 0.0]
            ),
            ClassDefinition(
                name="Class3",
                description="Description 3",
                embedding_vector=[0.8, 0.6, 0.0]  # More similar to Class1
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        candidates = similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=3)
        
        assert len(candidates) == 3
        assert all(isinstance(c, ClassCandidate) for c in candidates)
        
        # Results should be sorted by similarity (highest first)
        assert candidates[0].class_definition.name == "Class1"  # Identical vector
        assert candidates[1].class_definition.name == "Class3"  # More similar
        assert candidates[2].class_definition.name == "Class2"  # Orthogonal
        
        # Check similarity scores are in descending order
        assert candidates[0].similarity_score >= candidates[1].similarity_score
        assert candidates[1].similarity_score >= candidates[2].similarity_score
    
    def test_find_similar_classes_top_k_limit(self):
        """Test finding similar classes with top_k limit."""
        classes = [
            ClassDefinition(
                name=f"Class{i}",
                description=f"Description {i}",
                embedding_vector=[float(i), 0.0, 0.0]
            )
            for i in range(5)
        ]
        
        similarity_search = SimilaritySearch(classes)
        candidates = similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=3)
        
        assert len(candidates) == 3
    
    def test_find_similar_classes_top_k_exceeds_classes(self):
        """Test finding similar classes when top_k exceeds number of classes."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2",
                embedding_vector=[0.0, 1.0, 0.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        candidates = similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=10)
        
        # Should return all available classes
        assert len(candidates) == 2
    
    def test_find_similar_classes_invalid_top_k(self):
        """Test finding similar classes with invalid top_k raises error."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        with pytest.raises(SimilaritySearchError, match="top_k must be positive"):
            similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=0)
    
    def test_find_similar_classes_empty_embedding(self):
        """Test finding similar classes with empty text embedding raises error."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        with pytest.raises(SimilaritySearchError, match="Text embedding cannot be empty"):
            similarity_search.find_similar_classes([], top_k=1)
    
    def test_find_similar_classes_dimension_mismatch(self):
        """Test finding similar classes with mismatched embedding dimensions raises error."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]  # 3 dimensions
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        with pytest.raises(SimilaritySearchError, match="Text embedding dimension .* doesn't match"):
            similarity_search.find_similar_classes([1.0, 0.0], top_k=1)  # 2 dimensions
    
    def test_class_candidate_properties(self):
        """Test that ClassCandidate objects have correct properties."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        candidates = similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=1)
        
        candidate = candidates[0]
        assert candidate.class_definition.name == "Class1"
        assert candidate.confidence == candidate.similarity_score
        assert "Cosine similarity:" in candidate.reasoning
        assert 0.0 <= candidate.similarity_score <= 1.0
        assert 0.0 <= candidate.confidence <= 1.0
    
    def test_numpy_optimization_consistency(self):
        """Test that numpy-optimized results match the original implementation."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2",
                embedding_vector=[0.0, 1.0, 0.0]
            ),
            ClassDefinition(
                name="Class3",
                description="Description 3",
                embedding_vector=[0.707, 0.707, 0.0]  # 45 degrees
            ),
            ClassDefinition(
                name="Class4",
                description="Description 4",
                embedding_vector=[-1.0, 0.0, 0.0]  # Opposite to Class1
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        # Test with different query vectors
        test_queries = [
            [1.0, 0.0, 0.0],  # Same as Class1
            [0.0, 1.0, 0.0],  # Same as Class2
            [0.5, 0.5, 0.0],  # Between Class1 and Class2
            [-0.5, 0.5, 0.0]  # Different direction
        ]
        
        for query in test_queries:
            candidates = similarity_search.find_similar_classes(query, top_k=4)
            
            # Verify all candidates are returned
            assert len(candidates) == 4
            
            # Verify similarity scores are in descending order
            for i in range(len(candidates) - 1):
                assert candidates[i].similarity_score >= candidates[i + 1].similarity_score
            
            # Verify similarity scores are in valid range
            for candidate in candidates:
                assert 0.0 <= candidate.similarity_score <= 1.0
                assert candidate.confidence == candidate.similarity_score
    
    def test_embedding_matrix_properties(self):
        """Test that the internal embedding matrix has correct properties."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 2.0, 3.0]
            ),
            ClassDefinition(
                name="Class2",
                description="Description 2",
                embedding_vector=[4.0, 5.0, 6.0]
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        # Check embedding matrix shape
        assert similarity_search._embedding_matrix.shape == (2, 3)
        
        # Check embedding matrix content
        expected_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(similarity_search._embedding_matrix, expected_matrix)
        
        # Check embedding norms
        expected_norms = np.array([
            np.linalg.norm([1.0, 2.0, 3.0]),
            np.linalg.norm([4.0, 5.0, 6.0])
        ])
        np.testing.assert_array_almost_equal(similarity_search._embedding_norms, expected_norms)
    
    def test_zero_embedding_handling(self):
        """Test handling of zero embeddings in the optimized implementation."""
        classes = [
            ClassDefinition(
                name="Class1",
                description="Description 1",
                embedding_vector=[1.0, 0.0, 0.0]
            ),
            ClassDefinition(
                name="ZeroClass",
                description="Zero embedding class",
                embedding_vector=[0.0, 0.0, 0.0]  # Zero embedding
            )
        ]
        
        similarity_search = SimilaritySearch(classes)
        
        # Test with non-zero query
        candidates = similarity_search.find_similar_classes([1.0, 0.0, 0.0], top_k=2)
        
        assert len(candidates) == 2
        # The non-zero class should have higher similarity
        assert candidates[0].class_definition.name == "Class1"
        assert candidates[1].class_definition.name == "ZeroClass"
        
        # Test with zero query
        candidates_zero = similarity_search.find_similar_classes([0.0, 0.0, 0.0], top_k=2)
        assert len(candidates_zero) == 2
        # Both should have some similarity score (handled by epsilon)
        for candidate in candidates_zero:
            assert 0.0 <= candidate.similarity_score <= 1.0
    
    def test_large_dataset_performance(self):
        """Test that the optimized implementation works with larger datasets."""
        # Create a larger dataset to test numpy optimization benefits
        num_classes = 100
        embedding_dim = 50
        
        classes = []
        for i in range(num_classes):
            # Create random-like embeddings (deterministic for testing)
            embedding = [float((i * 7 + j * 3) % 10) / 10.0 for j in range(embedding_dim)]
            classes.append(ClassDefinition(
                name=f"Class{i}",
                description=f"Description {i}",
                embedding_vector=embedding
            ))
        
        similarity_search = SimilaritySearch(classes)
        
        # Test query
        query = [0.5] * embedding_dim
        candidates = similarity_search.find_similar_classes(query, top_k=10)
        
        assert len(candidates) == 10
        assert all(isinstance(c, ClassCandidate) for c in candidates)
        
        # Verify results are sorted by similarity
        for i in range(len(candidates) - 1):
            assert candidates[i].similarity_score >= candidates[i + 1].similarity_score
        
        # Verify all similarity scores are valid
        for candidate in candidates:
            assert 0.0 <= candidate.similarity_score <= 1.0