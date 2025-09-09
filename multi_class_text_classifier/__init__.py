"""
Multi-class text classifier using Strands Agents for large-scale classification.
"""

from .models import ClassDefinition, ClassCandidate, ClassificationResult, ClassifierConfig
from .text_classifier import TextClassifier
from .dataset_generator import DatasetGenerator
from .pdf_extractor import PDFExtractor, PDFContent, ImageContent, create_pdf_extractor
from .attribute_evaluator import LLMAttributeEvaluator
from .attribute_generator import AttributeGenerator
from .exceptions import (
    ClassifierError,
    InvalidInputError,
    ConfigurationError,
    ProcessingError,
    ConfidenceError
)

__version__ = "0.1.0"
__all__ = [
    "ClassDefinition",
    "ClassCandidate", 
    "ClassificationResult",
    "ClassifierConfig",
    "TextClassifier",
    "DatasetGenerator",
    "PDFExtractor",
    "PDFContent",
    "ImageContent",
    "create_pdf_extractor",
    "LLMAttributeEvaluator",
    "AttributeGenerator",
    "ClassifierError",
    "InvalidInputError",
    "ConfigurationError", 
    "ProcessingError",
    "ConfidenceError"
]