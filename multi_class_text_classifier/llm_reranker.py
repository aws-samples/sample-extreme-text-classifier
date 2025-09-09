"""
LLM-based reranking functionality for classification candidates.
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any
from strands import Agent
from strands.models import BedrockModel
from multi_class_text_classifier.models.data_models import (
    ClassCandidate, 
    RerankingConfig
)
from multi_class_text_classifier.exceptions import (
    RerankingError, 
    RerankingConfigError
)


logger = logging.getLogger(__name__)


class LLMReranker:
    """
    LLM-based reranker for classification candidates.
    
    Supports multiple reranking models including Nova Lite, Amazon Rerank, 
    and Cohere Rerank with configurable fallback behavior.
    """
    
    def __init__(self, config: RerankingConfig):
        """
        Initialize the LLM reranker with configuration.
        
        Args:
            config: Reranking configuration including model type and parameters
            
        Raises:
            RerankingConfigError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        
        # Initialize model-specific settings
        self._initialize_model_settings()
        
        logger.info(f"Initialized LLMReranker with model type: {config.model_type}")
    
    def _validate_config(self) -> None:
        """
        Validate the reranking configuration.
        
        Raises:
            RerankingConfigError: If configuration is invalid
        """
        try:
            # Basic validation is handled by RerankingConfig.__post_init__
            # Additional validation can be added here
            
            if self.config.model_parameters:
                self._validate_model_parameters()
                
        except ValueError as e:
            raise RerankingConfigError(f"Invalid reranking configuration: {e}")
    
    def _validate_model_parameters(self) -> None:
        """
        Validate model-specific parameters.
        
        Raises:
            RerankingConfigError: If model parameters are invalid
        """
        params = self.config.model_parameters or {}
        
        if self.config.model_type == "llm":
            # Validate generic LLM parameters for Strands
            if "temperature" in params and not (0.0 <= params["temperature"] <= 1.0):
                raise RerankingConfigError("LLM temperature must be between 0.0 and 1.0")
            if "max_tokens" in params and params["max_tokens"] <= 0:
                raise RerankingConfigError("LLM max_tokens must be positive")
                
        elif self.config.model_type == "amazon_rerank":
            # Validate Amazon Rerank parameters
            if "return_documents" in params and not isinstance(params["return_documents"], bool):
                raise RerankingConfigError("Amazon Rerank return_documents must be boolean")
                
        elif self.config.model_type == "cohere_rerank":
            # Validate Cohere Rerank parameters
            if "model" in params and not isinstance(params["model"], str):
                raise RerankingConfigError("Cohere Rerank model must be a string")
    
    def _initialize_model_settings(self) -> None:
        """Initialize model-specific settings and defaults."""
        self._model_defaults = {
            "llm": {
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 0.9
            },
            "amazon_rerank": {
                "return_documents": True,
                "max_chunks_per_query": 100
            },
            "cohere_rerank": {
                "model": "rerank-english-v2.0",
                "return_documents": True
            }
        }
        
        # Initialize Strands agent for LLM model type
        if self.config.model_type == "llm":
            self._initialize_strands_agent()
    
    def _initialize_strands_agent(self) -> None:
        """Initialize Strands agent for generic LLM reranking."""
        try:
            # Get model parameters with defaults
            params = self._get_model_parameters("llm")
            
            # Create Bedrock model with the specified model ID
            model = BedrockModel(
                model=self.config.model_id,
                params={
                    "max_tokens": params.get("max_tokens", 1000),
                    "temperature": params.get("temperature", 0.1),
                    "top_p": params.get("top_p", 0.9)
                }
            )
            
            # Import prompts
            from .prompts import RerankingPrompts
            
            # Create Strands agent for reranking
            self.agent = Agent(
                model=model,
                system_prompt=RerankingPrompts.system_prompt()
            )
            
            logger.info(f"Initialized Strands agent with model: {self.config.model_id}")
            
        except Exception as e:
            raise RerankingConfigError(f"Failed to initialize Strands agent: {e}")
    
    def rerank_candidates(
        self, 
        input_text: str, 
        candidates: List[ClassCandidate]
    ) -> List[ClassCandidate]:
        """
        Rerank classification candidates using LLM semantic understanding.
        
        Args:
            input_text: Original text being classified
            candidates: List of candidates from similarity search
            
        Returns:
            Reranked list of candidates with updated scores
            
        Raises:
            RerankingError: If reranking fails and fallback is disabled
        """
        if not input_text or not input_text.strip():
            raise RerankingError("Input text cannot be empty")
        
        if not candidates:
            logger.warning("No candidates provided for reranking")
            return candidates
        
        # Limit candidates to top_k_candidates
        candidates_to_rerank = candidates[:self.config.top_k_candidates]
        
        logger.info(
            f"Reranking {len(candidates_to_rerank)} candidates using {self.config.model_type}"
        )
        
        try:
            # Select and call the appropriate reranking method
            if self.config.model_type == "llm":
                reranked_candidates = self._call_llm(input_text, candidates_to_rerank)
            elif self.config.model_type == "amazon_rerank":
                reranked_candidates = self._call_amazon_rerank(input_text, candidates_to_rerank)
            elif self.config.model_type == "cohere_rerank":
                reranked_candidates = self._call_cohere_rerank(input_text, candidates_to_rerank)
            else:
                raise RerankingError(f"Unsupported model type: {self.config.model_type}")
            
            # Add any remaining candidates that weren't reranked
            remaining_candidates = candidates[self.config.top_k_candidates:]
            final_candidates = reranked_candidates + remaining_candidates
            
            logger.info(f"Successfully reranked {len(reranked_candidates)} candidates")
            return final_candidates
            
        except Exception as e:
            logger.error(f"Reranking failed with {self.config.model_type}: {e}")
            
            if self.config.fallback_on_error:
                logger.info("Falling back to original similarity search results")
                return candidates
            else:
                raise RerankingError(f"Reranking failed: {e}")
    
    def _call_llm(
        self, 
        input_text: str, 
        candidates: List[ClassCandidate]
    ) -> List[ClassCandidate]:
        """
        Call generic LLM model for reranking via Strands.
        
        Args:
            input_text: Original text being classified
            candidates: List of candidates to rerank
            
        Returns:
            Reranked candidates with LLM scores
            
        Raises:
            RerankingError: If LLM API call fails
        """
        logger.info(f"Calling LLM ({self.config.model_id}) for reranking")
        
        try:
            # Construct prompt for LLM reranking
            prompt = self._construct_llm_prompt(input_text, candidates)
            
            # Call Strands agent for reranking
            response = self.agent(prompt)
            response_text = response.message['content'][0]['text']
            
            logger.debug(f"LLM response: {response_text}")
            
            # Parse the LLM response to extract reranking scores
            reranked_candidates = self._parse_llm_response(response_text, candidates)
            
            logger.info(f"LLM reranking completed successfully for {len(reranked_candidates)} candidates")
            return reranked_candidates
            
        except Exception as e:
            raise RerankingError(f"LLM reranking failed: {e}")
    
    def _call_amazon_rerank(
        self, 
        input_text: str, 
        candidates: List[ClassCandidate]
    ) -> List[ClassCandidate]:
        """
        Call Amazon Rerank API for reranking.
        
        Args:
            input_text: Original text being classified
            candidates: List of candidates to rerank
            
        Returns:
            Reranked candidates with Amazon Rerank scores
            
        Raises:
            RerankingError: If Amazon Rerank API call fails
        """
        logger.info("Calling Amazon Rerank for reranking")
        
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
            from botocore.config import Config
            import time
            
            # Get model parameters with defaults
            params = self._get_model_parameters("amazon_rerank")
            
            # Import configuration
            from .config import config as classifier_config
            
            # Configure boto3 client with timeout and retry settings
            config = Config(
                region_name=self.config.aws_region or classifier_config.aws.bedrock_region,
                retries={
                    'max_attempts': classifier_config.retry.reranking_max_retries,
                    'mode': classifier_config.retry.mode
                },
                read_timeout=classifier_config.retry.reranking_read_timeout,
                connect_timeout=classifier_config.retry.reranking_connect_timeout
            )
            
            # Create bedrock agent runtime client
            client = boto3.client('bedrock-agent-runtime', config=config)
            
            # Prepare sources for reranking
            sources = []
            for candidate in candidates:
                # Create text document for each candidate
                source = {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {
                            "text": f"{candidate.class_definition.name}: {candidate.class_definition.description}"
                        }
                    }
                }
                sources.append(source)
            
            # Prepare the rerank request
            request_payload = {
                "queries": [
                    {
                        "type": "TEXT",
                        "textQuery": {
                            "text": input_text
                        }
                    }
                ],
                "sources": sources,
                "rerankingConfiguration": {
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {
                            "modelArn": self.config.model_id or "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
                        },
                        "numberOfResults": min(len(candidates), params.get("max_chunks_per_query", 100))
                    }
                }
            }
            
            # Add additional model request fields if specified
            if params.get("additional_model_request_fields"):
                request_payload["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]["additionalModelRequestFields"] = params["additional_model_request_fields"]
            
            logger.debug(f"Amazon Rerank request payload: {request_payload}")
            
            # Call Amazon Rerank API with retry logic
            max_retries = classifier_config.retry.reranking_max_retries
            base_delay = classifier_config.retry.reranking_base_delay
            
            for attempt in range(max_retries):
                try:
                    response = client.rerank(**request_payload)
                    break
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    
                    # Handle rate limiting
                    if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** attempt) + (time.time() % 1)
                            logger.warning(f"Rate limited by Amazon Rerank, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            raise RerankingError(f"Amazon Rerank rate limit exceeded after {max_retries} attempts: {e}")
                    
                    # Handle other client errors
                    elif error_code == 'ValidationException':
                        raise RerankingError(f"Amazon Rerank validation error: {e}")
                    elif error_code == 'AccessDeniedException':
                        raise RerankingError(f"Amazon Rerank access denied: {e}")
                    elif error_code == 'ResourceNotFoundException':
                        raise RerankingError(f"Amazon Rerank model not found: {e}")
                    else:
                        # For other client errors, retry if we have attempts left
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Amazon Rerank client error, retrying in {delay:.2f}s: {e}")
                            time.sleep(delay)
                            continue
                        else:
                            raise RerankingError(f"Amazon Rerank client error: {e}")
                
                except BotoCoreError as e:
                    # Handle connection and other boto3 errors
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Amazon Rerank connection error, retrying in {delay:.2f}s: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise RerankingError(f"Amazon Rerank connection error: {e}")
            
            logger.debug(f"Amazon Rerank response: {response}")
            
            # Process the response
            if 'results' not in response:
                raise RerankingError("Invalid response from Amazon Rerank: missing 'results' field")
            
            results = response['results']
            
            # Debug: Print the first result to see the actual structure
            if results:
                logger.info(f"Amazon Rerank first result structure: {results[0]}")
                logger.info(f"Amazon Rerank first result keys: {list(results[0].keys())}")
            
            # Create reranked candidates
            reranked_candidates = []
            
            # Create a mapping of original candidates by index
            candidate_map = {i: candidate for i, candidate in enumerate(candidates)}
            
            for result in results:
                # Get the original candidate index
                original_index = result.get('index')
                if original_index is None or original_index not in candidate_map:
                    logger.warning(f"Invalid index in Amazon Rerank result: {original_index}")
                    continue
                
                original_candidate = candidate_map[original_index]
                relevance_score = result.get('relevanceScore', 0.0)
                
                # Create new candidate with rerank score
                reranked_candidate = ClassCandidate(
                    class_definition=original_candidate.class_definition,
                    similarity_score=original_candidate.similarity_score,
                    rerank_score=float(relevance_score),
                    reasoning=f"{original_candidate.reasoning}; Amazon Rerank score: {relevance_score:.3f}",
                    attributes_matched=original_candidate.attributes_matched
                )
                reranked_candidates.append(reranked_candidate)
            
            # If we didn't get all candidates back, add missing ones with low scores
            reranked_indices = {result.get('index') for result in results if result.get('index') is not None}
            for i, original_candidate in enumerate(candidates):
                if i not in reranked_indices:
                    # Add with a low rerank score
                    fallback_candidate = ClassCandidate(
                        class_definition=original_candidate.class_definition,
                        similarity_score=original_candidate.similarity_score,
                        rerank_score=classifier_config.validation.default_fail_score,  # Low fallback score
                        reasoning=f"{original_candidate.reasoning}; Amazon Rerank: not returned",
                        attributes_matched=original_candidate.attributes_matched
                    )
                    reranked_candidates.append(fallback_candidate)
            
            # Sort by rerank score (highest first)
            reranked_candidates.sort(key=lambda c: c.rerank_score or 0, reverse=True)
            
            logger.info(f"Amazon Rerank reranking completed successfully for {len(reranked_candidates)} candidates")
            return reranked_candidates
            
        except ImportError as e:
            raise RerankingError(f"boto3 not available for Amazon Rerank: {e}")
        except Exception as e:
            raise RerankingError(f"Amazon Rerank reranking failed: {e}")
    
    def _call_cohere_rerank(
        self, 
        input_text: str, 
        candidates: List[ClassCandidate]
    ) -> List[ClassCandidate]:
        """
        Call Cohere Rerank API for reranking using Amazon Bedrock.
        
        Args:
            input_text: Original text being classified
            candidates: List of candidates to rerank
            
        Returns:
            Reranked candidates with Cohere Rerank scores
            
        Raises:
            RerankingError: If Cohere Rerank API call fails
        """
        logger.info("Calling Cohere Rerank for reranking")
        
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
            from botocore.config import Config
            import time
            import json
            
            # Get model parameters with defaults
            params = self._get_model_parameters("cohere_rerank")
            
            # Import configuration
            from .config import config as classifier_config
            
            # Configure boto3 client with timeout and retry settings (reused from Amazon Rerank)
            config = Config(
                region_name=self.config.aws_region or classifier_config.aws.bedrock_region,
                retries={
                    'max_attempts': classifier_config.retry.reranking_max_retries,
                    'mode': classifier_config.retry.mode
                },
                read_timeout=classifier_config.retry.reranking_read_timeout,
                connect_timeout=classifier_config.retry.reranking_connect_timeout
            )
            
            # Create bedrock runtime client for Cohere Rerank
            client = boto3.client('bedrock-runtime', config=config)
            
            # Prepare documents for Cohere Rerank
            documents = []
            for candidate in candidates:
                # Create text document for each candidate
                document_text = f"{candidate.class_definition.name}: {candidate.class_definition.description}"
                documents.append(document_text)
            
            # Prepare the Cohere Rerank request payload
            cohere_model_id = self.config.model_id or "cohere.rerank-v3-5:0"
            
            request_payload = {
                "api_version": 2,
                "query": input_text,
                "documents": documents
            }
            
            # Add optional parameters if provided
            if "top_n" in params:
                request_payload["top_n"] = min(len(candidates), params["top_n"])
            
            logger.debug(f"Cohere Rerank request payload: {request_payload}")
            
            # Call Cohere Rerank API with retry logic (reused from Amazon Rerank)
            max_retries = classifier_config.retry.reranking_max_retries
            base_delay = classifier_config.retry.reranking_base_delay
            
            for attempt in range(max_retries):
                try:
                    response = client.invoke_model(
                        modelId=cohere_model_id,
                        contentType='application/json',
                        accept='*/*',
                        body=json.dumps(request_payload)
                    )
                    
                    # Parse the response
                    response_body = json.loads(response['body'].read())
                    break
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    
                    # Handle rate limiting (reused error handling pattern from Amazon Rerank)
                    if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** attempt) + (time.time() % 1)
                            logger.warning(f"Rate limited by Cohere Rerank, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            raise RerankingError(f"Cohere Rerank rate limit exceeded after {max_retries} attempts: {e}")
                    
                    # Handle other client errors (reused error handling pattern from Amazon Rerank)
                    elif error_code == 'ValidationException':
                        raise RerankingError(f"Cohere Rerank validation error: {e}")
                    elif error_code == 'AccessDeniedException':
                        raise RerankingError(f"Cohere Rerank access denied: {e}")
                    elif error_code == 'ResourceNotFoundException':
                        raise RerankingError(f"Cohere Rerank model not found: {e}")
                    else:
                        # For other client errors, retry if we have attempts left
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Cohere Rerank client error, retrying in {delay:.2f}s: {e}")
                            time.sleep(delay)
                            continue
                        else:
                            raise RerankingError(f"Cohere Rerank client error: {e}")
                
                except BotoCoreError as e:
                    # Handle connection and other boto3 errors (reused from Amazon Rerank)
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Cohere Rerank connection error, retrying in {delay:.2f}s: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise RerankingError(f"Cohere Rerank connection error: {e}")
            
            logger.debug(f"Cohere Rerank response: {response_body}")
            
            # Process the Cohere Rerank response
            if 'results' not in response_body:
                raise RerankingError("Invalid response from Cohere Rerank: missing 'results' field")
            
            results = response_body['results']
            
            # Create reranked candidates
            reranked_candidates = []
            
            # Create a mapping of original candidates by index (reused pattern from Amazon Rerank)
            candidate_map = {i: candidate for i, candidate in enumerate(candidates)}
            
            for result in results:
                # Get the original candidate index
                original_index = result.get('index')
                if original_index is None or original_index not in candidate_map:
                    logger.warning(f"Invalid index in Cohere Rerank result: {original_index}")
                    continue
                
                original_candidate = candidate_map[original_index]
                relevance_score = result.get('relevance_score', 0.0)
                
                # Create new candidate with rerank score
                reranked_candidate = ClassCandidate(
                    class_definition=original_candidate.class_definition,
                    similarity_score=original_candidate.similarity_score,
                    rerank_score=float(relevance_score),
                    reasoning=f"{original_candidate.reasoning}; Cohere Rerank score: {relevance_score:.3f}",
                    attributes_matched=original_candidate.attributes_matched
                )
                reranked_candidates.append(reranked_candidate)
            
            # If we didn't get all candidates back, add missing ones with low scores (reused pattern from Amazon Rerank)
            reranked_indices = {result.get('index') for result in results if result.get('index') is not None}
            for i, original_candidate in enumerate(candidates):
                if i not in reranked_indices:
                    # Add with a low rerank score
                    fallback_candidate = ClassCandidate(
                        class_definition=original_candidate.class_definition,
                        similarity_score=original_candidate.similarity_score,
                        rerank_score=0.0,  # Low fallback score
                        reasoning=f"{original_candidate.reasoning}; Cohere Rerank: not returned",
                        attributes_matched=original_candidate.attributes_matched
                    )
                    reranked_candidates.append(fallback_candidate)
            
            # Sort by rerank score (highest first)
            reranked_candidates.sort(key=lambda c: c.rerank_score or 0, reverse=True)
            
            logger.info(f"Cohere Rerank reranking completed successfully for {len(reranked_candidates)} candidates")
            return reranked_candidates
            
        except ImportError as e:
            raise RerankingError(f"boto3 not available for Cohere Rerank: {e}")
        except Exception as e:
            raise RerankingError(f"Cohere Rerank reranking failed: {e}")
    
    def _get_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Get model parameters with defaults applied.
        
        Args:
            model_type: Type of model to get parameters for
            
        Returns:
            Dictionary of model parameters
        """
        defaults = self._model_defaults.get(model_type, {})
        user_params = self.config.model_parameters or {}
        
        # Merge defaults with user parameters (user parameters take precedence)
        return {**defaults, **user_params}
    
    def _construct_llm_prompt(
        self, 
        input_text: str, 
        candidates: List[ClassCandidate]
    ) -> str:
        """
        Construct a prompt for generic LLM reranking via Strands.
        
        Args:
            input_text: Original text being classified
            candidates: List of candidates to rerank
            
        Returns:
            Formatted prompt string for LLM
        """
        from .prompts import RerankingPrompts
        
        # Convert candidates to the format expected by the prompt
        candidate_dicts = [
            {
                'name': candidate.class_definition.name,
                'description': candidate.class_definition.description
            }
            for candidate in candidates
        ]
        
        return RerankingPrompts.reranking_prompt(input_text, candidate_dicts)
    
    def _parse_llm_response(
        self, 
        response_text: str, 
        original_candidates: List[ClassCandidate]
    ) -> List[ClassCandidate]:
        """
        Parse LLM response and create reranked candidates.
        
        Args:
            response_text: Raw response from LLM
            original_candidates: Original list of candidates
            
        Returns:
            List of reranked candidates with scores
            
        Raises:
            RerankingError: If parsing fails
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise RerankingError("No JSON found in LLM response")
            
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            
            if "rankings" not in parsed_data:
                raise RerankingError("Missing 'rankings' key in LLM response")
            
            rankings = parsed_data["rankings"]
            if not isinstance(rankings, list):
                raise RerankingError("'rankings' must be a list")
            
            # Create a mapping of class names to original candidates
            candidate_map = {
                candidate.class_definition.name: candidate 
                for candidate in original_candidates
            }
            
            reranked_candidates = []
            
            for ranking in rankings:
                if not isinstance(ranking, dict):
                    continue
                
                class_name = ranking.get("class_name", "").strip()
                score = ranking.get("score", 0.0)
                reasoning = ranking.get("reasoning", "")
                
                # Find the original candidate
                if class_name in candidate_map:
                    original_candidate = candidate_map[class_name]
                    
                    # Create new candidate with rerank score
                    reranked_candidate = ClassCandidate(
                        class_definition=original_candidate.class_definition,
                        similarity_score=original_candidate.similarity_score,
                        rerank_score=float(score),
                        reasoning=f"{original_candidate.reasoning}; LLM rerank: {reasoning}",
                        attributes_matched=original_candidate.attributes_matched
                    )
                    reranked_candidates.append(reranked_candidate)
            
            # If we didn't get all candidates back, add missing ones with low scores
            reranked_names = {c.class_definition.name for c in reranked_candidates}
            for original_candidate in original_candidates:
                if original_candidate.class_definition.name not in reranked_names:
                    # Add with a low rerank score
                    fallback_candidate = ClassCandidate(
                        class_definition=original_candidate.class_definition,
                        similarity_score=original_candidate.similarity_score,
                        rerank_score=0.1,  # Low fallback score
                        reasoning=f"{original_candidate.reasoning}; LLM rerank: fallback score",
                        attributes_matched=original_candidate.attributes_matched
                    )
                    reranked_candidates.append(fallback_candidate)
            
            # Sort by rerank score (highest first)
            reranked_candidates.sort(key=lambda c: c.rerank_score or 0, reverse=True)
            
            return reranked_candidates
            
        except json.JSONDecodeError as e:
            raise RerankingError(f"Failed to parse JSON from LLM response: {e}")
        except Exception as e:
            raise RerankingError(f"Failed to parse LLM response: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "model_type": self.config.model_type,
            "top_k_candidates": self.config.top_k_candidates,
            "fallback_on_error": self.config.fallback_on_error,
            "aws_region": self.config.aws_region,
            "has_api_key": bool(self.config.api_key),
            "model_parameters": self.config.model_parameters or {}
        }
    
    def validate_connection(self) -> bool:
        """
        Validate that the reranker can connect to the configured model.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # TODO: Implement actual connection validation for each model type
            # For now, just validate configuration
            self._validate_config()
            return True
        except RerankingConfigError:
            return False