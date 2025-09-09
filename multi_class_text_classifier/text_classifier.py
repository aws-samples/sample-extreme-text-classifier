"""
Main TextClassifier class for multi-class text classification.

This module provides the main TextClassifier class that integrates dataset loading,
embedding generation, and similarity search for text classification.
"""

from pathlib import Path
from typing import List, Optional
from .dataset_loader import ClassesDataset, DatasetLoadingError
from .similarity_search import SimilaritySearch, SimilaritySearchError
from .llm_reranker import LLMReranker
from .attribute_validator import AttributeValidator
from .models.data_models import (
    ClassDefinition, 
    ClassificationResult, 
    ClassCandidate, 
    RerankingConfig,
    AttributeValidationConfig,
    AttributeEvaluationConfig
)
from .exceptions import ClassifierError, RerankingError


class TextClassificationError(ClassifierError):
    """Raised when text classification operations fail."""
    pass


class TextClassifier:
    """
    Main text classifier class that integrates dataset loading, embedding generation,
    and similarity search for multi-class text classification.
    
    This class provides a simple interface for classifying text inputs using
    pre-trained embeddings and cosine similarity search.
    """
    
    def __init__(
        self, 
        dataset_path: Optional[str] = None, 
        embeddings_path: Optional[str] = None, 
        reranking_config: Optional[RerankingConfig] = None,
        attribute_config: Optional[AttributeValidationConfig] = None
    ):
        """
        Initialize the classifier with dataset and embeddings paths.
        
        Args:
            dataset_path: Optional path to JSON dataset file
            embeddings_path: Optional path to embeddings file
            reranking_config: Optional configuration for LLM-based reranking
            attribute_config: Optional configuration for attribute validation
            
        Note:
            - If both paths are provided, dataset_path is used for JSON loading and embeddings_path for embeddings
            - If only embeddings_path is provided, dataset is loaded from the embeddings file
            - If only dataset_path is provided, embeddings_path defaults to dataset_path with .pkl.gz extension
            - At least one path must be provided
            - If reranking_config is provided, reranking will be applied after similarity search
            - If attribute_config is provided and enabled, attribute validation will be applied after classification
            
        Raises:
            TextClassificationError: If initialization fails
        """
        if not dataset_path and not embeddings_path:
            raise TextClassificationError("At least one of dataset_path or embeddings_path must be provided")
        
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.reranking_config = reranking_config
        self.attribute_config = attribute_config
        
        # Track whether embeddings_path was explicitly provided
        self._embeddings_path_provided = embeddings_path is not None
        
        # Set default embeddings path if only dataset_path is provided
        if dataset_path and not embeddings_path:
            self.embeddings_path = self._get_default_embeddings_path(dataset_path)
        
        # Initialize components
        self._dataset: Optional[ClassesDataset] = None
        self._similarity_search: Optional[SimilaritySearch] = None
        self._reranker: Optional[LLMReranker] = None
        self._attribute_validator: Optional[AttributeValidator] = None
        self._classes: List[ClassDefinition] = []
        self._initialized = False
        
        # Initialize reranker if config is provided
        if self.reranking_config:
            try:
                self._reranker = LLMReranker(self.reranking_config)
            except (RerankingError, ValueError) as e:
                raise TextClassificationError(f"Failed to initialize reranker: {str(e)}")
        
        # Initialize attribute validator if config is provided and enabled
        if self.attribute_config and self.attribute_config.enabled:
            try:
                # Use default evaluation config if not provided
                eval_config = self.attribute_config.evaluation_config
                if eval_config is None:
                    eval_config = AttributeEvaluationConfig()
                
                self._attribute_validator = AttributeValidator(
                    attributes_path=self.attribute_config.attributes_path,
                    model_config=eval_config
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                raise TextClassificationError(f"Failed to initialize attribute validator: {str(e)}")
        
        # Validate paths
        self._validate_paths()
    
    def _get_default_embeddings_path(self, dataset_path: str) -> str:
        """
        Generate default embeddings path from dataset path.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Default embeddings path with .pkl.gz extension
        """
        dataset_path_obj = Path(dataset_path)
        return str(dataset_path_obj.with_suffix('.pkl.gz'))
    
    def _validate_paths(self) -> None:
        """
        Validate that the provided paths exist and are accessible.
        
        Raises:
            TextClassificationError: If paths are invalid
        """
        # Check dataset path if provided
        if self.dataset_path:
            dataset_path_obj = Path(self.dataset_path)
            if not dataset_path_obj.exists():
                raise TextClassificationError(f"Dataset file not found: {self.dataset_path}")
            
            if not dataset_path_obj.is_file():
                raise TextClassificationError(f"Dataset path is not a file: {self.dataset_path}")
        
        # Check embeddings path if provided
        if self.embeddings_path:
            embeddings_path_obj = Path(self.embeddings_path)
            # If only embeddings_path is provided, it must exist
            if not self.dataset_path and not embeddings_path_obj.exists():
                raise TextClassificationError(f"Embeddings file not found: {self.embeddings_path}")
            
            if embeddings_path_obj.exists() and not embeddings_path_obj.is_file():
                raise TextClassificationError(f"Embeddings path is not a file: {self.embeddings_path}")
    
    def _initialize_components(self) -> None:
        """
        Initialize dataset loader and similarity search components.
        
        This method loads the dataset, generates embeddings if needed, and
        sets up the similarity search component.
        
        Raises:
            TextClassificationError: If initialization fails
        """
        if self._initialized:
            return
        
        try:
            # Initialize dataset loader
            self._dataset = ClassesDataset(self.dataset_path)
            
            # If only embeddings_path is provided, load from embeddings file
            if not self.dataset_path and self.embeddings_path:
                embeddings_path_obj = Path(self.embeddings_path)
                if embeddings_path_obj.exists():
                    self._classes = self._dataset.load_dataset_with_embeddings(self.embeddings_path)
                else:
                    raise TextClassificationError(f"Embeddings file not found: {self.embeddings_path}")
            else:
                # Try to load embeddings first, fall back to generating them
                embeddings_path_obj = Path(self.embeddings_path)
                
                if embeddings_path_obj.exists():
                    # Load existing embeddings
                    try:
                        self._classes = self._dataset.load_dataset_with_embeddings(self.embeddings_path)
                    except DatasetLoadingError:
                        # If loading embeddings fails, fall back to JSON and generate embeddings
                        self._load_and_generate_embeddings()
                else:
                    # Generate embeddings from JSON dataset
                    self._load_and_generate_embeddings()
            
            # Initialize similarity search
            self._similarity_search = SimilaritySearch(self._classes)
            self._initialized = True
            
        except (DatasetLoadingError, SimilaritySearchError) as e:
            raise TextClassificationError(f"Failed to initialize classifier: {str(e)}")
        except Exception as e:
            raise TextClassificationError(f"Unexpected error during initialization: {str(e)}")
    
    def _load_and_generate_embeddings(self) -> None:
        """
        Load dataset from JSON and generate embeddings.
        
        Raises:
            TextClassificationError: If loading or embedding generation fails
        """
        try:
            # Load classes from JSON
            self._classes = self._dataset.load_classes_from_json()
            
            # Generate embeddings for classes that don't have them
            self._dataset.generate_embeddings()
            
            # Update classes with generated embeddings
            self._classes = self._dataset.classes
            
            # Only save embeddings if embeddings_path was explicitly provided
            if self._embeddings_path_provided:
                self._dataset.save_dataset_with_embeddings(self.embeddings_path)
            
        except DatasetLoadingError as e:
            raise TextClassificationError(f"Failed to load dataset and generate embeddings: {str(e)}")
    
    def predict(
        self, 
        text: str, 
        top_k: int = None, 
        retrieval_count: Optional[int] = None,
        validate_attributes: Optional[bool] = None
    ) -> ClassificationResult:
        """
        Classify a single text input using similarity search, optional reranking, and optional attribute validation.
        
        Args:
            text: Input text to classify
            top_k: Number of top candidates to consider for alternatives
            retrieval_count: Number of candidates to retrieve from similarity search before reranking.
                           If None, defaults to top_k. Should be >= top_k when reranking is enabled.
            validate_attributes: Override attribute validation configuration for this prediction.
                               If None, uses the configuration from initialization.
            
        Returns:
            ClassificationResult with prediction, confidence, alternatives, and optional attribute validation
            
        Raises:
            TextClassificationError: If classification fails
        """
        # Validate input
        from .config import config
        
        if not text or not text.strip():
            raise TextClassificationError("Input text cannot be empty")
        
        # Set default top_k if not provided
        if top_k is None:
            top_k = config.similarity.default_top_k
        
        if top_k <= 0:
            raise TextClassificationError("top_k must be positive")
        
        # Set default retrieval_count and validate
        if retrieval_count is None:
            retrieval_count = top_k
        
        if retrieval_count <= 0:
            raise TextClassificationError("retrieval_count must be positive")
        
        if retrieval_count < top_k:
            raise TextClassificationError("retrieval_count must be >= top_k")
        
        # Initialize components if needed
        self._initialize_components()
        
        try:
            # Generate embedding for input text
            text_embedding = self._dataset.generate_text_embedding(text.strip())
            
            # Find similar classes using similarity search
            candidates = self._similarity_search.find_similar_classes(text_embedding, retrieval_count)
            
            if not candidates:
                raise TextClassificationError("No similar classes found")
            
            # Apply reranking if configured
            reranked = False
            explanation_method = "cosine similarity"
            
            if self._reranker is not None:
                try:
                    # Apply LLM-based reranking to the candidates
                    candidates = self._reranker.rerank_candidates(text.strip(), candidates)
                    reranked = True
                    explanation_method = f"cosine similarity + {self.reranking_config.model_type} reranking"
                except RerankingError as e:
                    # If reranking fails and fallback is enabled, continue with similarity results
                    if self.reranking_config.fallback_on_error:
                        # Log the error but continue with similarity search results
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Reranking failed, falling back to similarity search: {e}")
                    else:
                        # If fallback is disabled, raise the error
                        raise TextClassificationError(f"Reranking failed: {str(e)}")
            
            # Trim candidates to top_k after reranking (if reranking was applied)
            if len(candidates) > top_k:
                candidates = candidates[:top_k]
            
            # Create classification result
            result = ClassificationResult.from_candidates(
                candidates=candidates,
                explanation=f"Classified using {explanation_method} with {len(self._classes)} classes",
                metadata={
                    "input_text": text,
                    "embedding_dimension": len(text_embedding),
                    "total_classes": len(self._classes),
                    "top_k": top_k,
                    "retrieval_count": retrieval_count,
                    "reranking_enabled": self._reranker is not None,
                    "reranking_applied": reranked,
                    "reranking_model": self.reranking_config.model_type if self.reranking_config else None,
                    "attribute_validation_configured": self._attribute_validator is not None
                }
            )
            
            # Set the reranked flag on the result
            result.reranked = reranked
            
            # Apply attribute validation if enabled
            should_validate = validate_attributes
            if should_validate is None:
                # Use configuration default
                should_validate = self._attribute_validator is not None
            
            if should_validate and self._attribute_validator is not None:
                try:
                    # Check if the class has attribute definitions
                    if self._attribute_validator.has_attributes_for_class(result.predicted_class.name):
                        # Validate attributes for the predicted class
                        attribute_result = self._attribute_validator.validate_prediction(
                            text=text.strip(),
                            predicted_class=result.predicted_class
                        )
                        result.attribute_validation = attribute_result
                        
                        # Update metadata to include attribute validation info
                        result.metadata["attribute_validation_enabled"] = True
                        result.metadata["attribute_score"] = attribute_result.overall_score
                    else:
                        # No attributes defined for this class - don't perform validation
                        result.metadata["attribute_validation_enabled"] = True
                        result.metadata["attribute_validation_skipped"] = True
                        result.metadata["attribute_validation_reason"] = f"No attribute definition found for class '{result.predicted_class.name}'"
                    
                except Exception as e:
                    # Log the error but don't fail the classification
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Attribute validation failed for class '{result.predicted_class.name}': {e}")
                    
                    # Add error info to metadata
                    result.metadata["attribute_validation_enabled"] = True
                    result.metadata["attribute_validation_error"] = str(e)
            else:
                # Add metadata to indicate attribute validation was not performed
                result.metadata["attribute_validation_enabled"] = False
            
            return result
            
        except (DatasetLoadingError, SimilaritySearchError) as e:
            raise TextClassificationError(f"Classification failed: {str(e)}")
        except Exception as e:
            raise TextClassificationError(f"Unexpected error during classification: {str(e)}")
    
    def get_class_count(self) -> int:
        """
        Get the number of classes available for classification.
        
        Returns:
            Number of classes in the dataset
            
        Raises:
            TextClassificationError: If classifier not initialized
        """
        self._initialize_components()
        return len(self._classes)
    
    def get_class_names(self) -> List[str]:
        """
        Get a list of all available class names.
        
        Returns:
            List of class names
            
        Raises:
            TextClassificationError: If classifier not initialized
        """
        self._initialize_components()
        return [class_def.name for class_def in self._classes]
    
    def get_class_by_name(self, name: str) -> ClassDefinition:
        """
        Get a specific class definition by name.
        
        Args:
            name: Name of the class to find
            
        Returns:
            ClassDefinition object
            
        Raises:
            TextClassificationError: If class not found or classifier not initialized
        """
        if not name or not name.strip():
            raise TextClassificationError("Class name cannot be empty")
        
        self._initialize_components()
        
        for class_def in self._classes:
            if class_def.name == name:
                return class_def
        
        raise TextClassificationError(f"Class not found: {name}")
    
    def get_embedding_info(self) -> dict:
        """
        Get information about the embeddings used by the classifier.
        
        Returns:
            Dictionary with embedding information
            
        Raises:
            TextClassificationError: If classifier not initialized
        """
        self._initialize_components()
        
        embedding_dim = None
        if self._classes and self._classes[0].embedding_vector:
            embedding_dim = len(self._classes[0].embedding_vector)
        
        return {
            "total_classes": len(self._classes),
            "embedding_dimension": embedding_dim,
            "embeddings_path": self.embeddings_path,
            "dataset_path": self.dataset_path,
            "classes_with_embeddings": sum(1 for c in self._classes if c.embedding_vector is not None)
        }
    
    def is_initialized(self) -> bool:
        """
        Check if the classifier has been initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
    
    def reinitialize(self) -> None:
        """
        Force reinitialization of the classifier components.
        
        This method can be used to reload the dataset and embeddings
        if they have been updated externally.
        
        Raises:
            TextClassificationError: If reinitialization fails
        """
        self._initialized = False
        self._dataset = None
        self._similarity_search = None
        self._classes = []
        
        # Reinitialize reranker if config is provided
        if self.reranking_config:
            try:
                self._reranker = LLMReranker(self.reranking_config)
            except (RerankingError, ValueError) as e:
                raise TextClassificationError(f"Failed to reinitialize reranker: {str(e)}")
        else:
            self._reranker = None
        
        # Reinitialize attribute validator if config is provided and enabled
        if self.attribute_config and self.attribute_config.enabled:
            try:
                # Use default evaluation config if not provided
                eval_config = self.attribute_config.evaluation_config
                if eval_config is None:
                    eval_config = AttributeEvaluationConfig()
                
                self._attribute_validator = AttributeValidator(
                    attributes_path=self.attribute_config.attributes_path,
                    model_config=eval_config
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                raise TextClassificationError(f"Failed to reinitialize attribute validator: {str(e)}")
        else:
            self._attribute_validator = None
        
        # Validate paths again
        self._validate_paths()
        
        # Initialize components
        self._initialize_components()