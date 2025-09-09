"""
Test project structure and basic interfaces.
"""

import pytest
from multi_class_text_classifier import (
    ClassDefinition,
    ClassCandidate,
    ClassificationResult,
    ClassifierConfig,
    TextClassifier,
    InvalidInputError,
    ConfigurationError
)


def test_class_definition_creation():
    """Test ClassDefinition can be created with valid inputs."""
    class_def = ClassDefinition(
        name="Electronics",
        description="Electronic devices and gadgets"
    )
    assert class_def.name == "Electronics"
    assert class_def.description == "Electronic devices and gadgets"
    assert class_def.embedding_vector is None
    assert class_def.metadata == {}


def test_class_definition_validation():
    """Test ClassDefinition validates inputs."""
    with pytest.raises(ValueError, match="Class name cannot be empty"):
        ClassDefinition(name="", description="Valid description")
    
    with pytest.raises(ValueError, match="Class description cannot be empty"):
        ClassDefinition(name="Valid name", description="")


def test_class_candidate_creation():
    """Test ClassCandidate can be created with valid inputs."""
    class_def = ClassDefinition(name="Test", description="Test class")
    candidate = ClassCandidate(
        class_definition=class_def,
        confidence=0.85,
        reasoning="High similarity match"
    )
    assert candidate.class_definition == class_def
    assert candidate.confidence == 0.85
    assert candidate.reasoning == "High similarity match"


def test_class_candidate_validation():
    """Test ClassCandidate validates confidence scores."""
    class_def = ClassDefinition(name="Test", description="Test class")
    
    with pytest.raises(ValueError, match="Confidence must be a number between 0.0 and 1.0"):
        ClassCandidate(class_definition=class_def, confidence=1.5)
    
    with pytest.raises(ValueError, match="Confidence must be a number between 0.0 and 1.0"):
        ClassCandidate(class_definition=class_def, confidence=-0.1)


def test_classifier_config_creation():
    """Test ClassifierConfig can be created with default values."""
    config = ClassifierConfig()
    assert config.enable_reranking is False
    assert config.enable_attribute_matching is False
    assert config.rag_retrieval_k == 50


def test_classifier_config_validation():
    """Test ClassifierConfig validates parameters."""
    with pytest.raises(ValueError, match="RAG retrieval k must be positive"):
        ClassifierConfig(rag_retrieval_k=0)


def test_text_classifier_initialization():
    """Test TextClassifier can be initialized with valid inputs."""
    classes = [
        ClassDefinition(name="Class1", description="First class"),
        ClassDefinition(name="Class2", description="Second class")
    ]
    config = ClassifierConfig()
    
    classifier = TextClassifier(classes=classes, config=config)
    assert len(classifier.classes) == 2
    assert classifier.config == config


def test_text_classifier_validation():
    """Test TextClassifier validates initialization inputs."""
    config = ClassifierConfig()
    
    # Test empty classes list
    with pytest.raises(InvalidInputError, match="Classes list cannot be empty"):
        TextClassifier(classes=[], config=config)
    
    # Test insufficient classes
    single_class = [ClassDefinition(name="Class1", description="First class")]
    with pytest.raises(InvalidInputError, match="At least 2 classes required"):
        TextClassifier(classes=single_class, config=config)
    
    # Test duplicate class names
    duplicate_classes = [
        ClassDefinition(name="Class1", description="First class"),
        ClassDefinition(name="Class1", description="Duplicate class")
    ]
    with pytest.raises(InvalidInputError, match="Duplicate class names found"):
        TextClassifier(classes=duplicate_classes, config=config)


def test_text_classifier_methods_not_implemented():
    """Test TextClassifier methods raise NotImplementedError for now."""
    from multi_class_text_classifier.models import TestSample
    
    classes = [
        ClassDefinition(name="Class1", description="First class"),
        ClassDefinition(name="Class2", description="Second class")
    ]
    config = ClassifierConfig()
    classifier = TextClassifier(classes=classes, config=config)
    
    with pytest.raises(NotImplementedError):
        classifier.predict("test text")
    
    with pytest.raises(NotImplementedError):
        classifier.generate_dataset()
    
    # Test with valid test dataset to reach NotImplementedError
    test_sample = TestSample(text="test", ground_truth_class="Class1")
    with pytest.raises(NotImplementedError):
        classifier.evaluate([test_sample])