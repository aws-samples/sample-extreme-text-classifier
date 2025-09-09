"""
Tests for the TextClassifier class.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from multi_class_text_classifier.text_classifier import TextClassifier, TextClassificationError
from multi_class_text_classifier.models.data_models import ClassDefinition, ClassificationResult


class TestTextClassifier:
    """Test cases for TextClassifier class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return {
            "classes": [
                {
                    "name": "Technology",
                    "description": "Articles about technology, software, and digital innovations"
                },
                {
                    "name": "Sports",
                    "description": "Content about sports, athletics, and competitions"
                },
                {
                    "name": "Health",
                    "description": "Health and medical information, wellness tips"
                }
            ]
        }
    
    @pytest.fixture
    def temp_dataset_file(self, sample_dataset):
        """Create a temporary dataset file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_dataset, f)
            return f.name
    
    def test_init_with_valid_paths(self, temp_dataset_file):
        """Test TextClassifier initialization with valid paths."""
        classifier = TextClassifier(temp_dataset_file)
        
        assert classifier.dataset_path == temp_dataset_file
        assert classifier.embeddings_path.endswith('.pkl.gz')
        assert not classifier.is_initialized()
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_init_with_custom_embeddings_path(self, temp_dataset_file):
        """Test TextClassifier initialization with custom embeddings path."""
        custom_embeddings_path = "/custom/path/embeddings.pkl.gz"
        classifier = TextClassifier(temp_dataset_file, custom_embeddings_path)
        
        assert classifier.dataset_path == temp_dataset_file
        assert classifier.embeddings_path == custom_embeddings_path
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_init_with_nonexistent_dataset(self):
        """Test TextClassifier initialization with nonexistent dataset file."""
        with pytest.raises(TextClassificationError, match="Dataset file not found"):
            TextClassifier("/nonexistent/dataset.json")
    
    def test_init_with_directory_as_dataset(self, tmp_path):
        """Test TextClassifier initialization with directory instead of file."""
        with pytest.raises(TextClassificationError, match="Dataset path is not a file"):
            TextClassifier(str(tmp_path))
    
    def test_get_default_embeddings_path(self, temp_dataset_file):
        """Test default embeddings path generation."""
        classifier = TextClassifier(temp_dataset_file)
        expected_path = str(Path(temp_dataset_file).with_suffix('.pkl.gz'))
        
        assert classifier.embeddings_path == expected_path
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_initialize_components_success(self, mock_dataset_class, temp_dataset_file):
        """Test successful component initialization."""
        # Mock dataset instance
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        # Mock classes with embeddings
        mock_classes = [
            ClassDefinition("Tech", "Technology content", [0.1, 0.2, 0.3]),
            ClassDefinition("Sports", "Sports content", [0.4, 0.5, 0.6])
        ]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        classifier._initialize_components()
        
        assert classifier.is_initialized()
        assert len(classifier._classes) == 2
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_classify_empty_text(self, temp_dataset_file):
        """Test classification with empty text input."""
        classifier = TextClassifier(temp_dataset_file)
        
        with pytest.raises(TextClassificationError, match="Input text cannot be empty"):
            classifier.predict("")
        
        with pytest.raises(TextClassificationError, match="Input text cannot be empty"):
            classifier.predict("   ")
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_classify_invalid_top_k(self, temp_dataset_file):
        """Test classification with invalid top_k parameter."""
        classifier = TextClassifier(temp_dataset_file)
        
        with pytest.raises(TextClassificationError, match="top_k must be positive"):
            classifier.predict("test text", top_k=0)
        
        with pytest.raises(TextClassificationError, match="top_k must be positive"):
            classifier.predict("test text", top_k=-1)
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_classify_invalid_retrieval_count(self, temp_dataset_file):
        """Test classification with invalid retrieval_count parameter."""
        classifier = TextClassifier(temp_dataset_file)
        
        with pytest.raises(TextClassificationError, match="retrieval_count must be positive"):
            classifier.predict("test text", retrieval_count=0)
        
        with pytest.raises(TextClassificationError, match="retrieval_count must be >= top_k"):
            classifier.predict("test text", top_k=5, retrieval_count=3)
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    @patch('multi_class_text_classifier.text_classifier.SimilaritySearch')
    def test_classify_success(self, mock_similarity_class, mock_dataset_class, temp_dataset_file):
        """Test successful text classification."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [
            ClassDefinition("Tech", "Technology content", [0.1, 0.2, 0.3]),
            ClassDefinition("Sports", "Sports content", [0.4, 0.5, 0.6])
        ]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        mock_dataset.generate_text_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock similarity search
        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity
        
        from multi_class_text_classifier.models.data_models import ClassCandidate
        mock_candidates = [
            ClassCandidate(mock_classes[0], similarity_score=0.8, reasoning="High similarity"),
            ClassCandidate(mock_classes[1], similarity_score=0.6, reasoning="Medium similarity")
        ]
        mock_similarity.find_similar_classes.return_value = mock_candidates
        
        classifier = TextClassifier(temp_dataset_file)
        result = classifier.predict("technology article")
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_class.name == "Tech"
        assert result.effective_score == 0.8
        assert len(result.alternatives) == 2  # All candidates become alternatives
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_get_class_count(self, mock_dataset_class, temp_dataset_file):
        """Test getting class count."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [
            ClassDefinition("Tech", "Technology", [0.1, 0.2]),
            ClassDefinition("Sports", "Sports", [0.3, 0.4]),
            ClassDefinition("Health", "Health", [0.5, 0.6])
        ]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        count = classifier.get_class_count()
        
        assert count == 3
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_get_class_names(self, mock_dataset_class, temp_dataset_file):
        """Test getting class names."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [
            ClassDefinition("Technology", "Tech content", [0.1, 0.2]),
            ClassDefinition("Sports", "Sports content", [0.3, 0.4])
        ]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        names = classifier.get_class_names()
        
        assert names == ["Technology", "Sports"]
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_get_class_by_name_success(self, mock_dataset_class, temp_dataset_file):
        """Test getting class by name successfully."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        tech_class = ClassDefinition("Technology", "Tech content", [0.1, 0.2])
        sports_class = ClassDefinition("Sports", "Sports content", [0.3, 0.4])
        mock_classes = [tech_class, sports_class]
        
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        found_class = classifier.get_class_by_name("Technology")
        
        assert found_class == tech_class
        assert found_class.name == "Technology"
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_get_class_by_name_not_found(self, mock_dataset_class, temp_dataset_file):
        """Test getting class by name when class doesn't exist."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [ClassDefinition("Technology", "Tech content", [0.1, 0.2])]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        
        with pytest.raises(TextClassificationError, match="Class not found: NonExistent"):
            classifier.get_class_by_name("NonExistent")
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    def test_get_class_by_name_empty_name(self, temp_dataset_file):
        """Test getting class by empty name."""
        classifier = TextClassifier(temp_dataset_file)
        
        with pytest.raises(TextClassificationError, match="Class name cannot be empty"):
            classifier.get_class_by_name("")
        
        with pytest.raises(TextClassificationError, match="Class name cannot be empty"):
            classifier.get_class_by_name("   ")
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_get_embedding_info(self, mock_dataset_class, temp_dataset_file):
        """Test getting embedding information."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [
            ClassDefinition("Tech", "Technology", [0.1, 0.2, 0.3]),
            ClassDefinition("Sports", "Sports", [0.4, 0.5, 0.6])
        ]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        info = classifier.get_embedding_info()
        
        assert info["total_classes"] == 2
        assert info["embedding_dimension"] == 3
        assert info["classes_with_embeddings"] == 2
        assert info["dataset_path"] == temp_dataset_file
        assert info["embeddings_path"].endswith('.pkl.gz')
        
        # Clean up
        Path(temp_dataset_file).unlink()
    
    @patch('multi_class_text_classifier.text_classifier.ClassesDataset')
    def test_reinitialize(self, mock_dataset_class, temp_dataset_file):
        """Test classifier reinitialization."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset
        
        mock_classes = [ClassDefinition("Tech", "Technology", [0.1, 0.2])]
        mock_dataset.load_classes_from_json.return_value = mock_classes
        mock_dataset.classes = mock_classes
        
        classifier = TextClassifier(temp_dataset_file)
        
        # Initialize first time
        classifier._initialize_components()
        assert classifier.is_initialized()
        
        # Reinitialize
        classifier.reinitialize()
        assert classifier.is_initialized()
        
        # Clean up
        Path(temp_dataset_file).unlink()