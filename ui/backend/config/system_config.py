"""
System configuration for the multi-class text classifier backend.
Contains AWS regions, model IDs, and other system-level settings.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AWSConfig:
    """AWS-related configuration settings."""
    bedrock_region: str = "us-west-2"
    
    # Model configurations
    default_nova_lite_model: str = "us.amazon.nova-lite-v1:0"
    default_nova_pro_model: str = "us.amazon.nova-pro-v1:0"
    default_claude_sonnet_4_model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    
    # Reranking model configurations
    amazon_rerank_model: str = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
    cohere_rerank_model: str = "cohere.rerank-v3-5:0"
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        """Create AWS config from environment variables."""
        return cls(
            bedrock_region=os.getenv('AWS_BEDROCK_REGION', cls.bedrock_region),
            default_nova_lite_model=os.getenv('AWS_NOVA_LITE_MODEL', cls.default_nova_lite_model),
            default_nova_pro_model=os.getenv('AWS_NOVA_PRO_MODEL', cls.default_nova_pro_model),
            default_claude_sonnet_4_model=os.getenv('AWS_CLAUDE_SONNET_4_MODEL', cls.default_claude_sonnet_4_model),
            amazon_rerank_model=os.getenv('AWS_AMAZON_RERANK_MODEL', cls.amazon_rerank_model),
            cohere_rerank_model=os.getenv('AWS_COHERE_RERANK_MODEL', cls.cohere_rerank_model),
        )


@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    # Default model parameters
    default_temperature: float = 0.1
    default_max_tokens: int = 4000
    
    # Attribute generation settings
    attribute_generation_temperature: float = 0.1
    attribute_generation_max_tokens: int = 4000
    
    # Attribute evaluation settings
    attribute_evaluation_temperature: float = 0.1
    attribute_evaluation_max_tokens: int = 1000
    
    # Example generation settings
    example_generation_temperature: float = 0.7
    example_generation_max_tokens: int = 4000
    
    # Reranking settings
    reranking_temperature: float = 0.1
    reranking_max_tokens: int = 1000
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create model config from environment variables."""
        return cls(
            default_temperature=float(os.getenv('MODEL_DEFAULT_TEMPERATURE', cls.default_temperature)),
            default_max_tokens=int(os.getenv('MODEL_DEFAULT_MAX_TOKENS', cls.default_max_tokens)),
            attribute_generation_temperature=float(os.getenv('MODEL_ATTR_GEN_TEMPERATURE', cls.attribute_generation_temperature)),
            attribute_generation_max_tokens=int(os.getenv('MODEL_ATTR_GEN_MAX_TOKENS', cls.attribute_generation_max_tokens)),
            attribute_evaluation_temperature=float(os.getenv('MODEL_ATTR_EVAL_TEMPERATURE', cls.attribute_evaluation_temperature)),
            attribute_evaluation_max_tokens=int(os.getenv('MODEL_ATTR_EVAL_MAX_TOKENS', cls.attribute_evaluation_max_tokens)),
            example_generation_temperature=float(os.getenv('MODEL_EXAMPLE_GEN_TEMPERATURE', cls.example_generation_temperature)),
            example_generation_max_tokens=int(os.getenv('MODEL_EXAMPLE_GEN_MAX_TOKENS', cls.example_generation_max_tokens)),
            reranking_temperature=float(os.getenv('MODEL_RERANKING_TEMPERATURE', cls.reranking_temperature)),
            reranking_max_tokens=int(os.getenv('MODEL_RERANKING_MAX_TOKENS', cls.reranking_max_tokens)),
        )


@dataclass
class APIConfig:
    """API-related configuration settings."""
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])
    max_file_size_mb: int = 50
    temp_file_cleanup_hours: int = 24
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        cors_origins = os.getenv('API_CORS_ORIGINS', ','.join(cls().cors_origins)).split(',')
        return cls(
            cors_origins=[origin.strip() for origin in cors_origins],
            max_file_size_mb=int(os.getenv('API_MAX_FILE_SIZE_MB', cls().max_file_size_mb)),
            temp_file_cleanup_hours=int(os.getenv('API_TEMP_FILE_CLEANUP_HOURS', cls().temp_file_cleanup_hours)),
        )


@dataclass
class RetryConfig:
    """Retry configuration for AWS services."""
    max_attempts: int = 10
    mode: str = "standard"
    read_timeout: int = 180
    connect_timeout: int = 15
    max_pool_connections: int = 5
    
    @classmethod
    def from_env(cls) -> 'RetryConfig':
        """Create retry config from environment variables."""
        return cls(
            max_attempts=int(os.getenv('AWS_RETRY_MAX_ATTEMPTS', cls.max_attempts)),
            mode=os.getenv('AWS_RETRY_MODE', cls.mode),
            read_timeout=int(os.getenv('AWS_READ_TIMEOUT', cls.read_timeout)),
            connect_timeout=int(os.getenv('AWS_CONNECT_TIMEOUT', cls.connect_timeout)),
            max_pool_connections=int(os.getenv('AWS_MAX_POOL_CONNECTIONS', cls.max_pool_connections)),
        )


@dataclass
class SystemConfig:
    """Main system configuration container."""
    aws: AWSConfig = field(default_factory=AWSConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create system config from environment variables."""
        return cls(
            aws=AWSConfig.from_env(),
            models=ModelConfig.from_env(),
            api=APIConfig.from_env(),
            retry=RetryConfig.from_env(),
        )


# Global configuration instance
config = SystemConfig.from_env()