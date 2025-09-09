"""
Configuration for the multi-class text classifier library.
Independent of UI backend configuration.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AWSConfig:
    """AWS-related configuration settings."""
    bedrock_region: str = "us-west-2"
    
    # Model configurations
    default_nova_lite_model: str = "us.amazon.nova-lite-v1:0"
    default_nova_pro_model: str = "us.amazon.nova-pro-v1:0"
    default_claude_sonnet_4_model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        """Create AWS config from environment variables."""
        return cls(
            bedrock_region=os.getenv('AWS_BEDROCK_REGION', cls.bedrock_region),
            default_nova_lite_model=os.getenv('AWS_NOVA_LITE_MODEL', cls.default_nova_lite_model),
            default_nova_pro_model=os.getenv('AWS_NOVA_PRO_MODEL', cls.default_nova_pro_model),
            default_claude_sonnet_4_model=os.getenv('AWS_CLAUDE_SONNET_4_MODEL', cls.default_claude_sonnet_4_model),
        )


@dataclass
class RetryConfig:
    """Retry configuration for AWS services."""
    max_attempts: int = 10
    mode: str = "standard"
    read_timeout: int = 180
    connect_timeout: int = 15
    max_pool_connections: int = 5
    
    # Reranking-specific retry settings
    reranking_max_retries: int = 3
    reranking_base_delay: float = 1.0
    reranking_read_timeout: int = 30
    reranking_connect_timeout: int = 10
    
    # Attribute generation retry settings
    attribute_generation_max_retries: int = 3
    attribute_generation_retry_delay: int = 5
    attribute_generation_class_delay: int = 3
    
    @classmethod
    def from_env(cls) -> 'RetryConfig':
        """Create retry config from environment variables."""
        return cls(
            max_attempts=int(os.getenv('AWS_RETRY_MAX_ATTEMPTS', cls.max_attempts)),
            mode=os.getenv('AWS_RETRY_MODE', cls.mode),
            read_timeout=int(os.getenv('AWS_READ_TIMEOUT', cls.read_timeout)),
            connect_timeout=int(os.getenv('AWS_CONNECT_TIMEOUT', cls.connect_timeout)),
            max_pool_connections=int(os.getenv('AWS_MAX_POOL_CONNECTIONS', cls.max_pool_connections)),
            reranking_max_retries=int(os.getenv('RERANKING_MAX_RETRIES', cls.reranking_max_retries)),
            reranking_base_delay=float(os.getenv('RERANKING_BASE_DELAY', cls.reranking_base_delay)),
            reranking_read_timeout=int(os.getenv('RERANKING_READ_TIMEOUT', cls.reranking_read_timeout)),
            reranking_connect_timeout=int(os.getenv('RERANKING_CONNECT_TIMEOUT', cls.reranking_connect_timeout)),
            attribute_generation_max_retries=int(os.getenv('ATTR_GEN_MAX_RETRIES', cls.attribute_generation_max_retries)),
            attribute_generation_retry_delay=int(os.getenv('ATTR_GEN_RETRY_DELAY', cls.attribute_generation_retry_delay)),
            attribute_generation_class_delay=int(os.getenv('ATTR_GEN_CLASS_DELAY', cls.attribute_generation_class_delay)),
        )


@dataclass
class PDFConfig:
    """Configuration for PDF extraction."""
    # Model configuration
    model_id: str = "us.amazon.nova-lite-v1:0"
    
    # Model parameters
    max_tokens: int = 4000
    temperature: float = 0.1
    top_p: float = 0.9
    
    # Image processing
    max_image_size: int = 1024
    image_quality: int = 85
    min_image_size: int = 50
    
    # Processing limits
    max_images_per_page: int = 10
    max_total_images: int = 50
    
    # Processing timeouts
    read_timeout: int = 60
    connect_timeout: int = 10
    
    # Default confidence score
    default_confidence_score: float = 0.8
    
    @classmethod
    def from_env(cls) -> 'PDFConfig':
        """Create PDF config from environment variables."""
        return cls(
            model_id=os.getenv('PDF_MODEL_ID', cls.model_id),
            max_tokens=int(os.getenv('PDF_MAX_TOKENS', cls.max_tokens)),
            temperature=float(os.getenv('PDF_TEMPERATURE', cls.temperature)),
            top_p=float(os.getenv('PDF_TOP_P', cls.top_p)),
            max_image_size=int(os.getenv('PDF_MAX_IMAGE_SIZE', cls.max_image_size)),
            image_quality=int(os.getenv('PDF_IMAGE_QUALITY', cls.image_quality)),
            min_image_size=int(os.getenv('PDF_MIN_IMAGE_SIZE', cls.min_image_size)),
            max_images_per_page=int(os.getenv('PDF_MAX_IMAGES_PER_PAGE', cls.max_images_per_page)),
            max_total_images=int(os.getenv('PDF_MAX_TOTAL_IMAGES', cls.max_total_images)),
            read_timeout=int(os.getenv('PDF_READ_TIMEOUT', cls.read_timeout)),
            connect_timeout=int(os.getenv('PDF_CONNECT_TIMEOUT', cls.connect_timeout)),
            default_confidence_score=float(os.getenv('PDF_DEFAULT_CONFIDENCE', cls.default_confidence_score)),
        )


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Model configuration
    model_id: str = "us.amazon.nova-lite-v1:0"
    max_tokens: int = 4000
    
    # Default parameters
    default_num_classes: int = 50
    
    # File formatting
    json_indent: int = 2
    
    @classmethod
    def from_env(cls) -> 'DatasetConfig':
        """Create dataset config from environment variables."""
        return cls(
            model_id=os.getenv('DATASET_MODEL_ID', cls.model_id),
            max_tokens=int(os.getenv('DATASET_MAX_TOKENS', cls.max_tokens)),
            default_num_classes=int(os.getenv('DATASET_DEFAULT_NUM_CLASSES', cls.default_num_classes)),
            json_indent=int(os.getenv('DATASET_JSON_INDENT', cls.json_indent)),
        )


@dataclass
class SimilarityConfig:
    """Configuration for similarity search."""
    # Default parameters
    default_top_k: int = 5
    
    # Numerical stability
    zero_norm_epsilon: float = 1e-8
    
    @classmethod
    def from_env(cls) -> 'SimilarityConfig':
        """Create similarity config from environment variables."""
        return cls(
            default_top_k=int(os.getenv('SIMILARITY_DEFAULT_TOP_K', cls.default_top_k)),
            zero_norm_epsilon=float(os.getenv('SIMILARITY_ZERO_NORM_EPSILON', cls.zero_norm_epsilon)),
        )


@dataclass
class ValidationConfig:
    """Configuration for attribute validation."""
    # Default scores
    default_pass_score: float = 1.0
    default_fail_score: float = 0.0
    fallback_rerank_score: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'ValidationConfig':
        """Create validation config from environment variables."""
        return cls(
            default_pass_score=float(os.getenv('VALIDATION_DEFAULT_PASS_SCORE', cls.default_pass_score)),
            default_fail_score=float(os.getenv('VALIDATION_DEFAULT_FAIL_SCORE', cls.default_fail_score)),
            fallback_rerank_score=float(os.getenv('VALIDATION_FALLBACK_RERANK_SCORE', cls.fallback_rerank_score)),
        )


@dataclass
class ClassifierConfig:
    """Configuration for the text classifier library."""
    aws: AWSConfig
    retry: RetryConfig
    pdf: PDFConfig
    dataset: DatasetConfig
    similarity: SimilarityConfig
    validation: ValidationConfig
    
    @classmethod
    def from_env(cls) -> 'ClassifierConfig':
        """Create classifier config from environment variables."""
        return cls(
            aws=AWSConfig.from_env(),
            retry=RetryConfig.from_env(),
            pdf=PDFConfig.from_env(),
            dataset=DatasetConfig.from_env(),
            similarity=SimilarityConfig.from_env(),
            validation=ValidationConfig.from_env(),
        )


# Global configuration instance
config = ClassifierConfig.from_env()


