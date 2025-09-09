"""
Exception classes for the multi-class text classifier.
"""


class ClassifierError(Exception):
    """Base exception for classifier errors."""
    pass


class InvalidInputError(ClassifierError):
    """Raised when input text or parameters are invalid."""
    pass


class ConfigurationError(ClassifierError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(ClassifierError):
    """Raised when classification processing fails."""
    pass


class ConfidenceError(ClassifierError):
    """Raised when confidence calculation fails."""
    pass


class RerankingError(ClassifierError):
    """Raised when reranking fails."""
    pass


class RerankingConfigError(ClassifierError):
    """Raised when reranking configuration is invalid."""
    pass