"""
Dataset loader for loading class definitions from JSON files.
"""

import json
import pickle
import gzip
from pathlib import Path
from typing import List, Dict, Any
import boto3
from strands import Agent
from .models.data_models import ClassDefinition
from .exceptions import ClassifierError


class DatasetLoadingError(ClassifierError):
    """Raised when dataset loading fails."""
    pass


class ClassesDataset:
    """Loads class definitions from JSON dataset files."""
    
    def __init__(self, filepath: str = None):
        """
        Initialize the dataset loader with a JSON file path.
        
        Args:
            filepath: Path to the JSON dataset file
        """
        
        self.filepath = filepath
        self._classes: List[ClassDefinition] = []
        self._metadata: Dict[str, Any] = {}
        self._loaded = False
    
    def load_classes_from_json(self, filepath: str = None) -> List[ClassDefinition]:
        """
        Load class definitions from JSON file.
        
        Args:
            filepath: Optional path to JSON file (uses instance filepath if not provided)
            
        Returns:
            List of ClassDefinition objects
            
        Raises:
            DatasetLoadingError: If loading or validation fails
        """
        file_path = filepath or self.filepath
        
        if not file_path:
            raise DatasetLoadingError("No filepath provided")
        
        # Validate file exists and is readable
        path = Path(file_path)
        if not path.exists():
            raise DatasetLoadingError(f"Dataset file not found: {file_path}")
        
        if not path.is_file():
            raise DatasetLoadingError(f"Path is not a file: {file_path}")
        
        try:
            # Load and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetLoadingError(f"Invalid JSON in dataset file: {str(e)}")
        except Exception as e:
            raise DatasetLoadingError(f"Failed to read dataset file: {str(e)}")
        
        # Validate JSON structure
        self._validate_dataset_structure(data)
        
        # Extract metadata
        self._metadata = data.get("metadata", {})
        
        # Convert to ClassDefinition objects
        classes_data = data["classes"]
        class_definitions = []
        
        for i, class_data in enumerate(classes_data):
            try:
                class_def = ClassDefinition(
                    name=class_data["name"],
                    description=class_data["description"],
                    embedding_vector=class_data.get("embedding_vector"),
                    metadata=class_data.get("metadata", {})
                )
                class_definitions.append(class_def)
            except Exception as e:
                raise DatasetLoadingError(f"Failed to create ClassDefinition for class {i}: {str(e)}")
        
        # Cache the loaded classes
        self._classes = class_definitions
        self._loaded = True
        
        return class_definitions
    
    def _validate_dataset_structure(self, data: Dict[str, Any]) -> None:
        """
        Validate the structure of the loaded JSON dataset.
        
        Args:
            data: Parsed JSON data
            
        Raises:
            DatasetLoadingError: If structure is invalid
        """
        if not isinstance(data, dict):
            raise DatasetLoadingError("Dataset must be a JSON object")
        
        if "classes" not in data:
            raise DatasetLoadingError("Dataset missing required 'classes' field")
        
        classes = data["classes"]
        if not isinstance(classes, list):
            raise DatasetLoadingError("'classes' field must be a list")
        
        if len(classes) == 0:
            raise DatasetLoadingError("Dataset must contain at least one class")
        
        # Validate each class definition
        for i, class_data in enumerate(classes):
            if not isinstance(class_data, dict):
                raise DatasetLoadingError(f"Class {i} must be a JSON object")
            
            if "name" not in class_data:
                raise DatasetLoadingError(f"Class {i} missing required 'name' field")
            
            if "description" not in class_data:
                raise DatasetLoadingError(f"Class {i} missing required 'description' field")
            
            name = class_data["name"]
            description = class_data["description"]
            
            if not isinstance(name, str) or not name.strip():
                raise DatasetLoadingError(f"Class {i} name must be a non-empty string")
            
            if not isinstance(description, str) or not description.strip():
                raise DatasetLoadingError(f"Class {i} description must be a non-empty string")
            
            # Validate optional embedding_vector if present
            if "embedding_vector" in class_data:
                embedding = class_data["embedding_vector"]
                if embedding is not None:
                    if not isinstance(embedding, list):
                        raise DatasetLoadingError(f"Class {i} embedding_vector must be a list or null")
                    
                    if not all(isinstance(x, (int, float)) for x in embedding):
                        raise DatasetLoadingError(f"Class {i} embedding_vector must contain only numbers")
    
    @property
    def classes(self) -> List[ClassDefinition]:
        """
        Get the loaded class definitions.
        
        Returns:
            List of ClassDefinition objects
            
        Raises:
            DatasetLoadingError: If dataset hasn't been loaded yet
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        return self._classes
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the dataset metadata.
        
        Returns:
            Dictionary containing dataset metadata
            
        Raises:
            DatasetLoadingError: If dataset hasn't been loaded yet
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        return self._metadata
    
    @property
    def num_classes(self) -> int:
        """
        Get the number of loaded classes.
        
        Returns:
            Number of classes in the dataset
            
        Raises:
            DatasetLoadingError: If dataset hasn't been loaded yet
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        return len(self._classes)
    
    def get_class_by_name(self, name: str) -> ClassDefinition:
        """
        Get a specific class definition by name.
        
        Args:
            name: Name of the class to find
            
        Returns:
            ClassDefinition object
            
        Raises:
            DatasetLoadingError: If class not found or dataset not loaded
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        if not name or not name.strip():
            raise DatasetLoadingError("Class name cannot be empty")
        
        for class_def in self._classes:
            if class_def.name == name:
                return class_def
        
        raise DatasetLoadingError(f"Class not found: {name}")
    
    def get_class_names(self) -> List[str]:
        """
        Get a list of all class names in the dataset.
        
        Returns:
            List of class names
            
        Raises:
            DatasetLoadingError: If dataset hasn't been loaded yet
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        return [class_def.name for class_def in self._classes]
    
    def generate_embeddings(self) -> None:
        """
        Generate embeddings for classes that don't have them yet.
        
        This method generates embeddings for class name concatenated with class description
        using the Strands Agent framework. Only generates embeddings for classes that 
        don't already have them.
        
        Raises:
            DatasetLoadingError: If dataset not loaded or embedding generation fails
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        # Find classes without embeddings
        classes_needing_embeddings = [
            class_def for class_def in self._classes 
            if class_def.embedding_vector is None
        ]
        
        if not classes_needing_embeddings:
            return  # All classes already have embeddings
        
        try:            
            # Generate embeddings for classes that need them
            for class_def in classes_needing_embeddings:
                # Concatenate class name and description
                text_to_embed = f"{class_def.name}: {class_def.description}"

                embedding_vector = self.generate_text_embedding(text_to_embed)

                # Put embedding in class definition
                class_def.embedding_vector = embedding_vector
                    
        except Exception as e:
            raise DatasetLoadingError(f"Failed to generate embeddings: {str(e)}")
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            DatasetLoadingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise DatasetLoadingError("Text cannot be empty")
        
        try:
            # Create boto3 client for embedding generation
            client = boto3.client("bedrock-runtime", region_name="us-east-1")
            
            model_id = "amazon.titan-embed-text-v2:0"

            # Create request payload
            native_request = {"inputText": text.strip()}
            request = json.dumps(native_request).encode("utf-8")

            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)

            # Decode the model's native response body.
            model_response = json.loads(response["body"].read())
            embedding_vector = model_response['embedding']

            return embedding_vector
                    
        except Exception as e:
            raise DatasetLoadingError(f"Failed to generate text embedding: {str(e)}")
    
    def save_dataset_with_embeddings(self, filepath: str) -> None:
        """
        Save the dataset with embeddings to a size-optimized file format.
        
        Uses gzip compression and pickle for efficient storage of embeddings.
        
        Args:
            filepath: Path to save the dataset file
            
        Raises:
            DatasetLoadingError: If dataset not loaded or save fails
        """
        if not self._loaded:
            raise DatasetLoadingError("Dataset not loaded. Call load_classes_from_json() first.")
        
        try:
            # Prepare data for saving
            save_data = {
                "metadata": self._metadata,
                "classes": []
            }
            
            # Convert classes to serializable format
            for class_def in self._classes:
                class_data = {
                    "name": class_def.name,
                    "description": class_def.description,
                    "embedding_vector": class_def.embedding_vector,
                    "metadata": class_def.metadata
                }
                save_data["classes"].append(class_data)
            
            # Save with gzip compression for size optimization
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            raise DatasetLoadingError(f"Failed to save dataset: {str(e)}")
    
    def load_dataset_with_embeddings(self, filepath: str) -> List[ClassDefinition]:
        """
        Load dataset with embeddings from a size-optimized file format.
        
        Loads from gzip-compressed pickle format.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            List of ClassDefinition objects with embeddings
            
        Raises:
            DatasetLoadingError: If loading fails
        """
        # Validate file exists
        path = Path(filepath)
        if not path.exists():
            raise DatasetLoadingError(f"Dataset file not found: {filepath}")
        
        if not path.is_file():
            raise DatasetLoadingError(f"Path is not a file: {filepath}")
        
        try:
            # Load from gzip-compressed pickle
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
                
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load dataset: {str(e)}")
        
        # Validate structure
        if not isinstance(data, dict) or "classes" not in data:
            raise DatasetLoadingError("Invalid dataset file format")
        
        # Extract metadata
        self._metadata = data.get("metadata", {})
        
        # Convert to ClassDefinition objects
        classes_data = data["classes"]
        class_definitions = []
        
        for i, class_data in enumerate(classes_data):
            try:
                class_def = ClassDefinition(
                    name=class_data["name"],
                    description=class_data["description"],
                    embedding_vector=class_data.get("embedding_vector"),
                    metadata=class_data.get("metadata", {})
                )
                class_definitions.append(class_def)
            except Exception as e:
                raise DatasetLoadingError(f"Failed to create ClassDefinition for class {i}: {str(e)}")
        
        # Cache the loaded classes
        self._classes = class_definitions
        self._loaded = True
        
        return class_definitions