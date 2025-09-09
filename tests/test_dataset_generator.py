"""
Tests for the DatasetGenerator class.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from multi_class_text_classifier import DatasetGenerator
from multi_class_text_classifier.dataset_generator import DatasetGenerationError


class TestDatasetGenerator:
    """Test cases for DatasetGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = DatasetGenerator()
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_dummy_dataset_default(self, mock_agent_class):
        """Test generating dataset with default parameters."""
        # Mock the agent response
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Artificial Intelligence", "description": "Content related to AI technologies and machine learning algorithms"},
            {"name": "Web Development", "description": "Articles about frontend and backend web development practices"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 2)
        
        assert "metadata" in dataset
        assert "classes" in dataset
        assert dataset["metadata"]["domain"] == "technology"
        assert dataset["metadata"]["num_classes"] == 2
        assert dataset["metadata"]["model"] == "Amazon Nova Lite"
        assert len(dataset["classes"]) == 2
        
        # Verify class structure
        for class_def in dataset["classes"]:
            assert "name" in class_def
            assert "description" in class_def
            assert isinstance(class_def["name"], str)
            assert isinstance(class_def["description"], str)
            assert len(class_def["name"]) > 0
            assert len(class_def["description"]) > 0
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_dummy_dataset_custom_domain(self, mock_agent_class):
        """Test generating dataset with custom domain."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Marketing Strategy", "description": "Business content focused on marketing strategies and campaigns"},
            {"name": "Sales Management", "description": "Professional guidance for sales team management and processes"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset(domain="business", num_classes=2)
        
        assert dataset["metadata"]["domain"] == "business"
        assert dataset["metadata"]["num_classes"] == 2
        assert len(dataset["classes"]) == 2
    
    def test_generate_dummy_dataset_empty_domain(self):
        """Test generating dataset with empty domain."""
        with pytest.raises(DatasetGenerationError, match="Domain cannot be empty"):
            self.generator.generate_dummy_dataset("")
        
        with pytest.raises(DatasetGenerationError, match="Domain cannot be empty"):
            self.generator.generate_dummy_dataset("   ")
    
    def test_generate_dummy_dataset_invalid_num_classes(self):
        """Test generating dataset with invalid number of classes."""
        with pytest.raises(DatasetGenerationError, match="Number of classes must be positive"):
            self.generator.generate_dummy_dataset("technology", 0)
        
        with pytest.raises(DatasetGenerationError, match="Number of classes must be positive"):
            self.generator.generate_dummy_dataset("technology", -5)
    
    def test_generate_dummy_dataset_too_many_classes(self):
        """Test generating dataset with too many classes."""
        with pytest.raises(DatasetGenerationError, match="Cannot generate more than 200 classes"):
            self.generator.generate_dummy_dataset("technology", 250)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_dummy_dataset_invalid_json_response(self, mock_agent_class):
        """Test handling invalid JSON response from LLM."""
        mock_agent = Mock()
        mock_agent.run.return_value = "This is not valid JSON"
        mock_agent_class.return_value = mock_agent
        
        with pytest.raises(DatasetGenerationError, match="No valid JSON found"):
            self.generator.generate_dummy_dataset("technology", 5)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_dummy_dataset_missing_classes_key(self, mock_agent_class):
        """Test handling response missing classes key."""
        mock_agent = Mock()
        mock_agent.run.return_value = '{"invalid": "structure"}'
        mock_agent_class.return_value = mock_agent
        
        with pytest.raises(DatasetGenerationError, match="Missing 'classes' key"):
            self.generator.generate_dummy_dataset("technology", 5)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_dummy_dataset_invalid_class_structure(self, mock_agent_class):
        """Test handling invalid class structure in response."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Valid Class", "description": "Valid description"},
            {"invalid": "missing required fields"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        with pytest.raises(DatasetGenerationError, match="missing required fields"):
            self.generator.generate_dummy_dataset("technology", 2)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_save_dataset_success(self, mock_agent_class):
        """Test successfully saving dataset to file."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Test Class", "description": "Test description"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.generator.save_dataset(dataset, tmp_path)
            
            # Verify file was created and contains correct data
            assert Path(tmp_path).exists()
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                loaded_dataset = json.load(f)
            
            assert loaded_dataset == dataset
            assert len(loaded_dataset["classes"]) == 1
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_save_dataset_creates_directory(self, mock_agent_class):
        """Test that save_dataset creates directories if they don't exist."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Test Class", "description": "Test description"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = Path(tmp_dir) / "nested" / "directory" / "dataset.json"
            
            self.generator.save_dataset(dataset, str(nested_path))
            
            assert nested_path.exists()
            
            with open(nested_path, 'r', encoding='utf-8') as f:
                loaded_dataset = json.load(f)
            
            assert loaded_dataset == dataset
    
    def test_save_dataset_empty_dataset(self):
        """Test saving empty dataset raises error."""
        with pytest.raises(DatasetGenerationError, match="Dataset cannot be empty"):
            self.generator.save_dataset({}, "test.json")
        
        with pytest.raises(DatasetGenerationError, match="Dataset cannot be empty"):
            self.generator.save_dataset(None, "test.json")
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_save_dataset_empty_filepath(self, mock_agent_class):
        """Test saving with empty filepath raises error."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Test Class", "description": "Test description"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 1)
        
        with pytest.raises(DatasetGenerationError, match="Filepath cannot be empty"):
            self.generator.save_dataset(dataset, "")
        
        with pytest.raises(DatasetGenerationError, match="Filepath cannot be empty"):
            self.generator.save_dataset(dataset, None)
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_dataset_classes_are_sorted(self, mock_agent_class):
        """Test that generated classes are sorted by name."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Zebra Class", "description": "Last alphabetically"},
            {"name": "Alpha Class", "description": "First alphabetically"},
            {"name": "Beta Class", "description": "Second alphabetically"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 3)
        
        class_names = [cls["name"] for cls in dataset["classes"]]
        expected_sorted = ["Alpha Class", "Beta Class", "Zebra Class"]
        
        assert class_names == expected_sorted
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_generate_and_save_dataset(self, mock_agent_class):
        """Test the combined generate and save operation."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Test Class", "description": "Test description"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            dataset = self.generator.generate_and_save_dataset("technology", tmp_path, 1)
            
            # Verify dataset was returned
            assert "metadata" in dataset
            assert "classes" in dataset
            assert len(dataset["classes"]) == 1
            
            # Verify file was created
            assert Path(tmp_path).exists()
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                loaded_dataset = json.load(f)
            
            assert loaded_dataset == dataset
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_create_generation_prompt(self):
        """Test the prompt creation method."""
        prompt = self.generator._create_generation_prompt("technology", 10)
        
        assert "technology" in prompt
        assert "10" in prompt
        assert "JSON" in prompt
        assert "classes" in prompt
        assert "name" in prompt
        assert "description" in prompt
    
    @patch('multi_class_text_classifier.dataset_generator.Agent')
    def test_parse_llm_response_truncates_excess_classes(self, mock_agent_class):
        """Test that parser truncates if LLM returns more classes than requested."""
        mock_agent = Mock()
        mock_agent.run.return_value = '''
        {
          "classes": [
            {"name": "Class 1", "description": "Description 1"},
            {"name": "Class 2", "description": "Description 2"},
            {"name": "Class 3", "description": "Description 3"}
          ]
        }
        '''
        mock_agent_class.return_value = mock_agent
        
        dataset = self.generator.generate_dummy_dataset("technology", 2)
        
        # Should only have 2 classes even though LLM returned 3
        assert len(dataset["classes"]) == 2
        assert dataset["metadata"]["num_classes"] == 2