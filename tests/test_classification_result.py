"""
Tests for classification result structures and functionality.
"""

import pytest
from multi_class_text_classifier.models.data_models import (
    ClassDefinition, 
    ClassCandidate, 
    ClassificationResult
)


class TestClassificationResult:
    """Test cases for ClassificationResult class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample class definitions
        self.class1 = ClassDefinition(
            name="Brake Pads",
            description="Friction material that helps stop the vehicle when brakes are applied",
            metadata={"category": "braking"}
        )
        
        self.class2 = ClassDefinition(
            name="Windshield Wipers",
            description="System that removes rain and debris from the windshield",
            metadata={"category": "visibility"}
        )
        
        self.class3 = ClassDefinition(
            name="Battery",
            description="Electrical component that provides power to start the engine",
            metadata={"category": "electrical"}
        )
        
        # Create sample candidates
        self.candidate1 = ClassCandidate(
            class_definition=self.class1,
            similarity_score=0.85,
            reasoning="High similarity to braking system description"
        )
        
        self.candidate2 = ClassCandidate(
            class_definition=self.class2,
            similarity_score=0.72,
            reasoning="Moderate similarity to visibility system"
        )
        
        self.candidate3 = ClassCandidate(
            class_definition=self.class3,
            similarity_score=0.45,
            reasoning="Low similarity to electrical system"
        )
    
    def test_classification_result_creation(self):
        """Test basic ClassificationResult creation."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate2, self.candidate3],
            explanation="Test classification"
        )
        
        assert result.predicted_class == self.class1
        assert len(result.alternatives) == 2
        assert result.explanation == "Test classification"
    
    def test_classification_result_validation(self):
        """Test ClassificationResult validation."""
        # Test that basic creation works without validation errors
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1],
            explanation="Test"
        )
        assert result.predicted_class == self.class1
    
    def test_from_candidates_basic(self):
        """Test creating ClassificationResult from candidates."""
        candidates = [self.candidate1, self.candidate2, self.candidate3]
        
        result = ClassificationResult.from_candidates(candidates)
        
        # Should pick the highest effective score candidate as prediction
        assert result.predicted_class == self.class1
        assert result.effective_score == 0.85
        assert len(result.alternatives) == 3  # All candidates become alternatives
        assert result.alternatives[0].class_definition == self.class1  # Top candidate first
        assert result.alternatives[1].class_definition == self.class2
        assert result.alternatives[2].class_definition == self.class3
    
    def test_from_candidates_sorting(self):
        """Test creating ClassificationResult with proper sorting."""
        # Create candidates with different scores
        low_score_candidate = ClassCandidate(
            class_definition=self.class1,
            similarity_score=0.3,
            reasoning="Low score match"
        )
        
        candidates = [low_score_candidate, self.candidate2, self.candidate3]
        
        result = ClassificationResult.from_candidates(candidates)
        
        assert result.predicted_class == self.class2  # Highest score (0.72)
        assert result.effective_score == 0.72
        assert result.alternatives[0].class_definition == self.class2  # Top candidate first
    
    def test_from_candidates_empty_list(self):
        """Test creating ClassificationResult from empty candidates list."""
        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            ClassificationResult.from_candidates([])
    
    def test_from_candidates_unsorted_input(self):
        """Test that candidates are properly sorted by effective score."""
        # Create unsorted candidates
        unsorted_candidates = [self.candidate3, self.candidate1, self.candidate2]  # 0.45, 0.85, 0.72
        
        result = ClassificationResult.from_candidates(unsorted_candidates)
        
        # Should pick highest effective score as prediction
        assert result.predicted_class == self.class1  # 0.85 score
        assert result.effective_score == 0.85
        
        # Alternatives should be sorted by effective score (descending)
        assert result.alternatives[0].effective_score == 0.85  # class1
        assert result.alternatives[1].effective_score == 0.72  # class2
        assert result.alternatives[2].effective_score == 0.45  # class3
    
    def test_generate_explanation(self):
        """Test automatic explanation generation."""
        candidates = [self.candidate1, self.candidate2, self.candidate3]
        
        result = ClassificationResult.from_candidates(candidates)
        
        # Check that explanation contains key information
        assert "Brake Pads" in result.explanation
        assert "85.0%" in result.explanation
        assert "Windshield Wipers" in result.explanation  # Alternative
        assert "72.0%" in result.explanation
    
    def test_generate_explanation_single_candidate(self):
        """Test explanation generation for single candidate."""
        candidates = [self.candidate1]
        
        result = ClassificationResult.from_candidates(candidates)
        
        assert "Brake Pads" in result.explanation
        assert "85.0%" in result.explanation
        # Should not contain uncertainty warning since we removed that feature
    
    def test_get_top_candidates(self):
        """Test getting top candidates including prediction."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3]
        )
        
        # Test including prediction
        top_candidates = result.get_top_candidates(include_prediction=True, max_candidates=3)
        
        assert len(top_candidates) == 3
        assert top_candidates[0].class_definition == self.class1  # Prediction first
        assert top_candidates[0].effective_score == 0.85
        assert top_candidates[1].class_definition == self.class2  # First alternative
        assert top_candidates[2].class_definition == self.class3  # Second alternative
    
    def test_get_top_candidates_without_prediction(self):
        """Test getting top candidates without including prediction."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3]
        )
        
        top_candidates = result.get_top_candidates(include_prediction=False, max_candidates=5)
        
        assert len(top_candidates) == 2  # Alternatives excluding prediction
        assert top_candidates[0].class_definition == self.class2
        assert top_candidates[1].class_definition == self.class3
    
    def test_get_top_candidates_limit(self):
        """Test limiting the number of top candidates."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3]
        )
        
        top_candidates = result.get_top_candidates(include_prediction=True, max_candidates=2)
        
        assert len(top_candidates) == 2  # Limited to 2
        assert top_candidates[0].class_definition == self.class1  # Prediction
        assert top_candidates[1].class_definition == self.class2  # Top alternative
    
    def test_format_result_basic(self):
        """Test basic result formatting."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3],
            explanation="Test classification result"
        )
        
        formatted = result.format_result()
        
        # Check that key information is present
        assert "🎯 Classification Result" in formatted
        assert "Brake Pads" in formatted
        assert "85.0%" in formatted
        assert "Friction material that helps stop" in formatted
        assert "Test classification result" in formatted
        assert "Windshield Wipers" in formatted  # Alternative
        assert "Battery" in formatted  # Alternative
    
    def test_format_result_with_metadata(self):
        """Test formatting results with metadata."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1],
            metadata={"test": "value", "source": "test"}
        )
        
        formatted = result.format_result()
        
        assert "📊 Metadata: 2 items" in formatted
    
    def test_format_result_without_alternatives(self):
        """Test formatting results without alternatives."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[]
        )
        
        formatted = result.format_result(include_alternatives=True)
        
        # Should not include alternatives section
        assert "🔄 Top" not in formatted
        assert "Windshield Wipers" not in formatted
    
    def test_format_result_limit_alternatives(self):
        """Test limiting alternatives in formatting."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3]
        )
        
        formatted = result.format_result(include_alternatives=True, max_alternatives=1)
        
        # Should only show 1 alternative
        assert "🔄 Top 1 Alternatives:" in formatted
        assert "Brake Pads" in formatted  # First alternative (which is the prediction)
        # Should not show other alternatives beyond the limit
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[self.candidate1, self.candidate2, self.candidate3],
            explanation="Test explanation",
            metadata={"test": "value"}
        )
        
        result_dict = result.to_dict()
        
        # Check structure
        assert "predicted_class" in result_dict
        assert "effective_score" in result_dict
        assert "explanation" in result_dict
        assert "alternatives" in result_dict
        assert "metadata" in result_dict
        assert "reranked" in result_dict
        
        # Check predicted class structure
        predicted = result_dict["predicted_class"]
        assert predicted["name"] == "Brake Pads"
        assert predicted["description"] == "Friction material that helps stop the vehicle when brakes are applied"
        assert predicted["metadata"] == {"category": "braking"}
        
        # Check alternatives structure
        alternatives = result_dict["alternatives"]
        assert len(alternatives) == 3
        assert alternatives[0]["name"] == "Brake Pads"
        assert alternatives[0]["effective_score"] == 0.85
        assert alternatives[0]["similarity_score"] == 0.85
        assert alternatives[1]["name"] == "Windshield Wipers"
        assert alternatives[1]["effective_score"] == 0.72
        
        # Check other fields
        assert result_dict["effective_score"] == 0.85
        assert result_dict["explanation"] == "Test explanation"
        assert result_dict["metadata"] == {"test": "value"}
        assert result_dict["reranked"] is False
    
    def test_get_confidence_level(self):
        """Test confidence level categorization."""
        # Create candidates with different scores for testing
        high_candidate = ClassCandidate(class_definition=self.class1, similarity_score=0.9)
        medium_candidate = ClassCandidate(class_definition=self.class1, similarity_score=0.7)
        low_candidate = ClassCandidate(class_definition=self.class1, similarity_score=0.5)
        very_low_candidate = ClassCandidate(class_definition=self.class1, similarity_score=0.2)
        
        # Test High confidence
        result_high = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[high_candidate]
        )
        assert result_high.get_confidence_level() == "High"
        
        # Test Medium confidence
        result_medium = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[medium_candidate]
        )
        assert result_medium.get_confidence_level() == "Medium"
        
        # Test Low confidence
        result_low = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[low_candidate]
        )
        assert result_low.get_confidence_level() == "Low"
        
        # Test Very Low confidence
        result_very_low = ClassificationResult(
            predicted_class=self.class1,
            alternatives=[very_low_candidate]
        )
        assert result_very_low.get_confidence_level() == "Very Low"
    
    def test_custom_explanation(self):
        """Test using custom explanation in from_candidates."""
        candidates = [self.candidate1, self.candidate2]
        custom_explanation = "Custom explanation for this classification"
        
        result = ClassificationResult.from_candidates(
            candidates, 
            explanation=custom_explanation
        )
        
        assert result.explanation == custom_explanation
    
    def test_custom_metadata(self):
        """Test using custom metadata in from_candidates."""
        candidates = [self.candidate1, self.candidate2]
        custom_metadata = {"source": "test", "timestamp": "2024-01-01"}
        
        result = ClassificationResult.from_candidates(
            candidates,
            metadata=custom_metadata
        )
        
        assert result.metadata == custom_metadata
    
    def test_single_candidate(self):
        """Test creating result from single candidate."""
        candidates = [self.candidate1]
        
        result = ClassificationResult.from_candidates(candidates)
        
        assert result.predicted_class == self.class1
        assert result.effective_score == 0.85
        assert len(result.alternatives) == 1  # Single candidate becomes alternative
    
    def test_effective_score_calculation_consistency(self):
        """Test that effective score calculation is consistent across methods."""
        candidates = [self.candidate1, self.candidate2, self.candidate3]
        
        result = ClassificationResult.from_candidates(candidates)
        
        # Effective score should match the top candidate's effective score
        assert result.effective_score == self.candidate1.effective_score
        
        # Top candidates should maintain score consistency
        top_candidates = result.get_top_candidates(include_prediction=True)
        assert top_candidates[0].effective_score == result.effective_score