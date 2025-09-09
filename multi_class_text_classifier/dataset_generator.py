"""
Dataset generator for creating dummy classification datasets using LLM.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from strands import Agent
from strands.models import BedrockModel
from .exceptions import ClassifierError


class DatasetGenerationError(ClassifierError):
    """Raised when dataset generation fails."""
    pass


class DatasetGenerator:
    """Generates dummy datasets for text classification using Amazon Nova Lite."""
    
    def __init__(self):
        """Initialize the dataset generator with Nova Lite model."""
        from .config import config
        
        model = BedrockModel(
            model=config.dataset.model_id,
            params={
                "max_tokens": config.dataset.max_tokens
            }
        )

        from .prompts import DatasetGenerationPrompts
        
        self.agent = Agent(
            model=model,
            system_prompt=DatasetGenerationPrompts.system_prompt(),
        )
    
    def generate_dummy_dataset(self, domain: str, num_classes: int = None) -> Dict[str, Any]:
        """
        Generate a static JSON dataset with class definitions using LLM.
        
        Args:
            domain: Domain for the dataset (e.g., "technology", "business", "healthcare")
            num_classes: Number of classes to generate (defaults to config value)
            
        Returns:
            Dictionary containing the generated dataset
            
        Raises:
            DatasetGenerationError: If generation fails
        """
        from .config import config
        
        if num_classes is None:
            num_classes = config.dataset.default_num_classes
        """
        Generate a static JSON dataset with class definitions using LLM.
        
        Args:
            domain: Domain for the dataset (e.g., "technology", "business", "healthcare")
            num_classes: Number of classes to generate (default: 50)
            
        Returns:
            Dictionary containing class definitions with names and descriptions
            
        Raises:
            DatasetGenerationError: If generation fails
        """
        if not domain or not domain.strip():
            raise DatasetGenerationError("Domain cannot be empty")
        
        if num_classes <= 0:
            raise DatasetGenerationError("Number of classes must be positive")
        
        if num_classes > 200:
            raise DatasetGenerationError("Cannot generate more than 200 classes at once")
        
        from .prompts import DatasetGenerationPrompts
        prompt = DatasetGenerationPrompts.dataset_generation_prompt(domain, num_classes)
        response = self.agent(prompt)
        
        response_text = response.message['content'][0]['text']

        print(f"Response text: {response_text}")
        
        # Parse the LLM response to extract class definitions
        dataset = self._parse_llm_response(response_text, domain, num_classes)
        
        return dataset
    

    
    def _parse_llm_response(self, response: str, domain: str, num_classes: int) -> Dict[str, Any]:
        """
        Parse the LLM response and create a properly formatted dataset.
        
        Args:
            response: Raw response from the LLM
            domain: Domain for the dataset
            num_classes: Expected number of classes
            
        Returns:
            Formatted dataset dictionary
            
        Raises:
            DatasetGenerationError: If parsing fails
        """
        # Try to extract JSON from the response
        try:
            # Find the first { and last } to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                raise DatasetGenerationError("No valid JSON found in LLM response")
            
            json_str = response[start_idx:end_idx + 1]
        except Exception as e:
            raise DatasetGenerationError(f"Failed to extract JSON from response: {str(e)}")
                 
        parsed_data = json.loads(json_str)
        
        # Validate the parsed data
        if "classes" not in parsed_data:
            raise DatasetGenerationError("Missing 'classes' key in LLM response")
        
        classes = parsed_data["classes"]
        if not isinstance(classes, list):
            raise DatasetGenerationError("'classes' must be a list")
                    
        # Validate each class definition
        validated_classes = []
        for i, class_def in enumerate(classes):
            if not isinstance(class_def, dict):
                raise DatasetGenerationError(f"Class {i} is not a dictionary")
            
            if "name" not in class_def or "description" not in class_def:
                raise DatasetGenerationError(f"Class {i} missing required fields")
            
            name = class_def["name"].strip()
            description = class_def["description"].strip()
            
            if not name or not description:
                raise DatasetGenerationError(f"Class {i} has empty name or description")
            
            validated_classes.append({
                "name": name,
                "description": description
            })
        
        # Sort classes by name for consistency
        validated_classes.sort(key=lambda x: x["name"])
        
        # Create final dataset structure
        dataset = {
            "metadata": {
                "domain": domain,
                "num_classes": len(validated_classes),
                "generated_by": "DatasetGenerator",
                "model": "Amazon Nova Lite",
                "version": "1.0"
            },
            "classes": validated_classes
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            dataset: Dataset dictionary to save
            filepath: Path to save the JSON file
            
        Raises:
            DatasetGenerationError: If saving fails
        """
        if not dataset:
            raise DatasetGenerationError("Dataset cannot be empty")
        
        if not filepath:
            raise DatasetGenerationError("Filepath cannot be empty")
        
        try:
            # Create directory if it doesn't exist
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save dataset with proper formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=config.dataset.json_indent, ensure_ascii=False)
                
        except Exception as e:
            raise DatasetGenerationError(f"Failed to save dataset to {filepath}: {str(e)}")
    
    def generate_and_save_dataset(self, domain: str, filepath: str, num_classes: int = None) -> Dict[str, Any]:
        """
        Generate a dataset and save it to file in one operation.
        
        Args:
            domain: Domain for the dataset
            filepath: Path to save the JSON file
            num_classes: Number of classes to generate (default: 50)
            
        Returns:
            Generated dataset dictionary
            
        Raises:
            DatasetGenerationError: If generation or saving fails
        """
        dataset = self.generate_dummy_dataset(domain, num_classes)
        self.save_dataset(dataset, filepath)
        return dataset