"""
Tests for the dataset loader functionality.
"""

import json
import pytest
import tempfile
import gzip
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from multi_class_text_classifier.dataset_loader import ClassesDataset, DatasetLoadingError
from multi_class_text_classifier.models.data_models import ClassDefinition


class TestClassesDataset:
    """Test cases for ClassesDataset class."""
    
    def test_init_with_valid_filepath(self):
        """Test initialization with valid filepath."""
        dataset = ClassesDataset("test.json")
        assert dataset.filepath == "test.json"
        assert not dataset._loaded
    
    def test_init_with_empty_filepath(self):
        """Test initialization with empty filepath raises error."""
        with pytest.raises(DatasetLoadingError, match="Filepath cannot be empty"):
            ClassesDataset("")
        
        with pytest.raises(DatasetLoadingError, match="Filepath cannot be empty"):
            ClassesDataset("   ")
    
    def test_load_classes_from_valid_json(self):
        """Test loading classes from valid JSON file."""
        # Create test dataset
        test_data = {
            "metadata": {
                "domain": "test",
                "num_classes": 2,
                "version": "1.0"
            },
            "classes": [
                {
                    "name": "Technology",
                    "description": "Articles about technology and software"
                },
                {
                    "name": "Sports",
                    "description": "Content about sports and athletics"
                }
            ]
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test loading
            dataset = ClassesDataset(temp_path)
            classes = dataset.load_classes_from_json()
            
            # Verify results
            assert len(classes) == 2
            assert isinstance(classes[0], ClassDefinition)
            assert isinstance(classes[1], ClassDefinition)
            
            assert classes[0].name == "Technology"
            assert classes[0].description == "Articles about technology and software"
            assert classes[0].embedding_vector is None
            
            assert classes[1].name == "Sports"
            assert classes[1].description == "Content about sports and athletics"
            assert classes[1].embedding_vector is None
            
            # Test properties
            assert dataset.num_classes == 2
            assert dataset.metadata["domain"] == "test"
            assert dataset.metadata["num_classes"] == 2
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_load_classes_with_embeddings(self):
        """Test loading classes with embedding vectors."""
        test_data = {
            "classes": [
                {
                    "name": "Test Class",
                    "description": "A test class",
                    "embedding_vector": [0.1, 0.2, 0.3]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            classes = dataset.load_classes_from_json()
            
            assert len(classes) == 1
            assert classes[0].embedding_vector == [0.1, 0.2, 0.3]
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        dataset = ClassesDataset("nonexistent.json")
        
        with pytest.raises(DatasetLoadingError, match="Dataset file not found"):
            dataset.load_classes_from_json()
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            
            with pytest.raises(DatasetLoadingError, match="Invalid JSON"):
                dataset.load_classes_from_json()
                
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_classes_field(self):
        """Test loading JSON without classes field raises error."""
        test_data = {"metadata": {"domain": "test"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            
            with pytest.raises(DatasetLoadingError, match="missing required 'classes' field"):
                dataset.load_classes_from_json()
                
        finally:
            Path(temp_path).unlink()
    
    def test_load_empty_classes_list(self):
        """Test loading with empty classes list raises error."""
        test_data = {"classes": []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            
            with pytest.raises(DatasetLoadingError, match="must contain at least one class"):
                dataset.load_classes_from_json()
                
        finally:
            Path(temp_path).unlink()
    
    def test_load_invalid_class_structure(self):
        """Test loading with invalid class structure raises error."""
        test_data = {
            "classes": [
                {
                    "name": "Valid Class",
                    "description": "Valid description"
                },
                {
                    "name": "",  # Invalid empty name
                    "description": "Valid description"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            
            with pytest.raises(DatasetLoadingError, match="name must be a non-empty string"):
                dataset.load_classes_from_json()
                
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_required_fields(self):
        """Test loading with missing required fields raises error."""
        test_data = {
            "classes": [
                {
                    "name": "Test Class"
                    # Missing description
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            
            with pytest.raises(DatasetLoadingError, match="missing required 'description' field"):
                dataset.load_classes_from_json()
                
        finally:
            Path(temp_path).unlink()
    
    def test_get_class_by_name(self):
        """Test getting class by name."""
        test_data = {
            "classes": [
                {
                    "name": "Technology",
                    "description": "Tech content"
                },
                {
                    "name": "Sports",
                    "description": "Sports content"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Test finding existing class
            tech_class = dataset.get_class_by_name("Technology")
            assert tech_class.name == "Technology"
            assert tech_class.description == "Tech content"
            
            # Test class not found
            with pytest.raises(DatasetLoadingError, match="Class not found: NonExistent"):
                dataset.get_class_by_name("NonExistent")
                
        finally:
            Path(temp_path).unlink()
    
    def test_get_class_names(self):
        """Test getting list of class names."""
        test_data = {
            "classes": [
                {"name": "A", "description": "First"},
                {"name": "B", "description": "Second"},
                {"name": "C", "description": "Third"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            names = dataset.get_class_names()
            assert names == ["A", "B", "C"]
            
        finally:
            Path(temp_path).unlink()
    
    def test_properties_before_loading(self):
        """Test that properties raise error before loading."""
        dataset = ClassesDataset("test.json")
        
        with pytest.raises(DatasetLoadingError, match="Dataset not loaded"):
            _ = dataset.classes
            
        with pytest.raises(DatasetLoadingError, match="Dataset not loaded"):
            _ = dataset.metadata
            
        with pytest.raises(DatasetLoadingError, match="Dataset not loaded"):
            _ = dataset.num_classes
    
    def test_load_with_alternative_filepath(self):
        """Test loading with alternative filepath parameter."""
        test_data = {"classes": [{"name": "Test", "description": "Test class"}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Initialize with different path, but load from temp_path
            dataset = ClassesDataset("different.json")
            classes = dataset.load_classes_from_json(temp_path)
            
            assert len(classes) == 1
            assert classes[0].name == "Test"
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_existing_car_parts_dataset(self):
        """Test loading the existing car parts dataset."""
        dataset_path = "datasets/car_parts_dataset.json"
        
        # Check if file exists (skip test if not)
        if not Path(dataset_path).exists():
            pytest.skip(f"Dataset file not found: {dataset_path}")
        
        dataset = ClassesDataset(dataset_path)
        classes = dataset.load_classes_from_json()
        
        # Verify basic structure
        assert len(classes) > 0
        assert all(isinstance(c, ClassDefinition) for c in classes)
        assert all(c.name and c.description for c in classes)
        
        # Verify metadata
        metadata = dataset.metadata
        assert "domain" in metadata
        assert metadata["domain"] == "car parts"
        assert metadata["num_classes"] == len(classes)


class TestEmbeddingGeneration:
    """Test cases for embedding generation functionality."""
    
    def test_generate_embeddings_not_loaded(self):
        """Test that generate_embeddings raises error when dataset not loaded."""
        dataset = ClassesDataset("test.json")
        
        with pytest.raises(DatasetLoadingError, match="Dataset not loaded"):
            dataset.generate_embeddings()
    
    def test_generate_embeddings_all_have_embeddings(self):
        """Test generate_embeddings when all classes already have embeddings."""
        test_data = {
            "classes": [
                {
                    "name": "Test Class",
                    "description": "A test class",
                    "embedding_vector": [0.1, 0.2, 0.3]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Should not raise error and should not change embeddings
            original_embedding = dataset.classes[0].embedding_vector.copy()
            dataset.generate_embeddings()
            
            assert dataset.classes[0].embedding_vector == original_embedding
            
        finally:
            Path(temp_path).unlink()
    
    @patch('multi_class_text_classifier.dataset_loader.boto3')
    def test_generate_embeddings_with_bedrock_success(self, mock_boto3):
        """Test successful embedding generation using Bedrock."""
        # Mock boto3 client and response
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        # Create a 1024-dimensional embedding
        mock_embedding = [0.1] * 1024
        mock_response = {
            "body": MagicMock()
        }
        mock_response["body"].read.return_value = json.dumps({
            "embedding": mock_embedding
        }).encode()
        mock_client.invoke_model.return_value = mock_response
        
        test_data = {
            "classes": [
                {
                    "name": "Technology",
                    "description": "Tech content"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Verify no embedding initially
            assert dataset.classes[0].embedding_vector is None
            
            # Generate embeddings
            dataset.generate_embeddings()
            
            # Verify embedding was generated
            assert dataset.classes[0].embedding_vector is not None
            assert len(dataset.classes[0].embedding_vector) == 1024
            assert all(isinstance(x, (int, float)) for x in dataset.classes[0].embedding_vector)
            
            # Verify boto3 client was called correctly
            mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
            mock_client.invoke_model.assert_called_once()
            
        finally:
            Path(temp_path).unlink()
    
    @patch('multi_class_text_classifier.dataset_loader.boto3')
    def test_generate_embeddings_with_bedrock_failure(self, mock_boto3):
        """Test embedding generation failure handling."""
        # Mock boto3 client to raise an exception
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Bedrock API error")
        
        test_data = {
            "classes": [
                {
                    "name": "Technology",
                    "description": "Tech content"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Generate embeddings should fail
            with pytest.raises(DatasetLoadingError, match="Failed to generate embeddings"):
                dataset.generate_embeddings()
            
        finally:
            Path(temp_path).unlink()
    
    def test_generate_embeddings_mixed_classes(self):
        """Test generating embeddings for mixed classes (some with, some without)."""
        test_data = {
            "classes": [
                {
                    "name": "Has Embedding",
                    "description": "Already has embedding",
                    "embedding_vector": [0.1, 0.2, 0.3]
                },
                {
                    "name": "No Embedding",
                    "description": "Needs embedding"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Store original embedding
            original_embedding = dataset.classes[0].embedding_vector.copy()
            
            # Generate embeddings
            dataset.generate_embeddings()
            
            # Verify first class embedding unchanged
            assert dataset.classes[0].embedding_vector == original_embedding
            
            # Verify second class got embedding
            assert dataset.classes[1].embedding_vector is not None
            assert len(dataset.classes[1].embedding_vector) == 1024
            
        finally:
            Path(temp_path).unlink()


class TestDatasetSaveLoad:
    """Test cases for dataset save/load with embeddings."""
    
    def test_save_dataset_not_loaded(self):
        """Test that save_dataset_with_embeddings raises error when dataset not loaded."""
        dataset = ClassesDataset("test.json")
        
        with pytest.raises(DatasetLoadingError, match="Dataset not loaded"):
            dataset.save_dataset_with_embeddings("output.pkl.gz")
    
    def test_save_and_load_dataset_with_embeddings(self):
        """Test saving and loading dataset with embeddings."""
        # Create test data with embeddings
        test_data = {
            "metadata": {
                "domain": "test",
                "version": "1.0"
            },
            "classes": [
                {
                    "name": "Technology",
                    "description": "Tech content",
                    "embedding_vector": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {"category": "tech"}
                },
                {
                    "name": "Sports",
                    "description": "Sports content",
                    "embedding_vector": [0.5, 0.6, 0.7, 0.8],
                    "metadata": {"category": "sports"}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as f:
            pkl_path = f.name
        
        try:
            # Load original dataset
            dataset = ClassesDataset(json_path)
            dataset.load_classes_from_json()
            
            # Save with embeddings
            dataset.save_dataset_with_embeddings(pkl_path)
            
            # Verify file was created and has content
            assert Path(pkl_path).exists()
            assert Path(pkl_path).stat().st_size > 0
            
            # Load from saved file
            new_dataset = ClassesDataset("dummy.json")
            loaded_classes = new_dataset.load_dataset_with_embeddings(pkl_path)
            
            # Verify loaded data matches original
            assert len(loaded_classes) == 2
            assert loaded_classes[0].name == "Technology"
            assert loaded_classes[0].description == "Tech content"
            assert loaded_classes[0].embedding_vector == [0.1, 0.2, 0.3, 0.4]
            assert loaded_classes[0].metadata == {"category": "tech"}
            
            assert loaded_classes[1].name == "Sports"
            assert loaded_classes[1].description == "Sports content"
            assert loaded_classes[1].embedding_vector == [0.5, 0.6, 0.7, 0.8]
            assert loaded_classes[1].metadata == {"category": "sports"}
            
            # Verify metadata
            assert new_dataset.metadata["domain"] == "test"
            assert new_dataset.metadata["version"] == "1.0"
            
        finally:
            Path(json_path).unlink()
            Path(pkl_path).unlink()
    
    def test_load_dataset_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        dataset = ClassesDataset("test.json")
        
        with pytest.raises(DatasetLoadingError, match="Dataset file not found"):
            dataset.load_dataset_with_embeddings("nonexistent.pkl.gz")
    
    def test_load_dataset_invalid_file(self):
        """Test loading from invalid file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("invalid content")
            temp_path = f.name
        
        try:
            dataset = ClassesDataset("test.json")
            
            with pytest.raises(DatasetLoadingError, match="Failed to load dataset"):
                dataset.load_dataset_with_embeddings(temp_path)
                
        finally:
            Path(temp_path).unlink()
    
    def test_save_dataset_invalid_path(self):
        """Test saving to invalid path raises error."""
        test_data = {"classes": [{"name": "Test", "description": "Test class"}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            dataset = ClassesDataset(temp_path)
            dataset.load_classes_from_json()
            
            # Try to save to invalid path (directory that doesn't exist)
            invalid_path = "/nonexistent/directory/file.pkl.gz"
            
            with pytest.raises(DatasetLoadingError, match="Failed to save dataset"):
                dataset.save_dataset_with_embeddings(invalid_path)
                
        finally:
            Path(temp_path).unlink()
    
    def test_compression_efficiency(self):
        """Test that compressed format is more efficient than JSON."""
        # Create dataset with large embeddings
        large_embedding = [0.1] * 1024  # 1024-dimensional embedding
        test_data = {
            "classes": [
                {
                    "name": f"Class_{i}",
                    "description": f"Description for class {i}",
                    "embedding_vector": large_embedding
                }
                for i in range(10)  # 10 classes with large embeddings
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as f:
            pkl_path = f.name
        
        try:
            # Load and save
            dataset = ClassesDataset(json_path)
            dataset.load_classes_from_json()
            dataset.save_dataset_with_embeddings(pkl_path)
            
            # Compare file sizes
            json_size = Path(json_path).stat().st_size
            pkl_size = Path(pkl_path).stat().st_size
            
            # Compressed file should be smaller (or at least not much larger)
            # This is a rough check - exact compression depends on data
            assert pkl_size < json_size * 2  # Allow some overhead but expect compression
            
        finally:
            Path(json_path).unlink()
            Path(pkl_path).unlink()