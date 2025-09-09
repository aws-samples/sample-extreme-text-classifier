"""
Tests for the LLMReranker class.
"""

import pytest
from unittest.mock import Mock, patch
from multi_class_text_classifier.llm_reranker import LLMReranker
from multi_class_text_classifier.models.data_models import (
    ClassDefinition, 
    ClassCandidate, 
    RerankingConfig
)
from multi_class_text_classifier.exceptions import (
    RerankingError, 
    RerankingConfigError
)


class TestLLMReranker:
    """Test cases for LLMReranker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test class definitions
        self.class_def_1 = ClassDefinition(
            name="Technology",
            description="Articles about technology and software"
        )
        self.class_def_2 = ClassDefinition(
            name="Sports",
            description="Articles about sports and athletics"
        )
        self.class_def_3 = ClassDefinition(
            name="Business",
            description="Articles about business and finance"
        )
        
        # Create test candidates
        self.candidates = [
            ClassCandidate(
                class_definition=self.class_def_1,
                confidence=0.8,
                similarity_score=0.8,
                reasoning="High similarity to technology content"
            ),
            ClassCandidate(
                class_definition=self.class_def_2,
                confidence=0.6,
                similarity_score=0.6,
                reasoning="Moderate similarity to sports content"
            ),
            ClassCandidate(
                class_definition=self.class_def_3,
                confidence=0.4,
                similarity_score=0.4,
                reasoning="Low similarity to business content"
            )
        ]
        
        # Create test input text
        self.input_text = "Latest developments in artificial intelligence and machine learning"
    
    def test_init_with_valid_llm_config(self):
        """Test initialization with valid LLM configuration."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            top_k_candidates=5
        )
        
        reranker = LLMReranker(config)
        assert reranker.config == config
        assert reranker.config.model_type == "llm"
        assert hasattr(reranker, 'agent')  # Should have Strands agent initialized
    
    def test_init_with_valid_amazon_rerank_config(self):
        """Test initialization with valid Amazon Rerank configuration."""
        config = RerankingConfig(
            model_type="amazon_rerank",
            api_key="test-api-key",
            top_k_candidates=3
        )
        
        reranker = LLMReranker(config)
        assert reranker.config == config
        assert reranker.config.model_type == "amazon_rerank"
    
    def test_init_with_valid_cohere_rerank_config(self):
        """Test initialization with valid Cohere Rerank configuration."""
        config = RerankingConfig(
            model_type="cohere_rerank",
            api_key="test-cohere-key",
            top_k_candidates=10
        )
        
        reranker = LLMReranker(config)
        assert reranker.config == config
        assert reranker.config.model_type == "cohere_rerank"
    
    def test_init_with_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError):
            config = RerankingConfig(
                model_type="invalid_model",
                api_key="test-key"
            )
            LLMReranker(config)
    
    def test_init_with_missing_model_id_for_llm(self):
        """Test initialization with missing model ID for LLM."""
        with pytest.raises(ValueError):
            config = RerankingConfig(
                model_type="llm"
                # Missing model_id
            )
            LLMReranker(config)
    
    def test_init_with_missing_api_key_for_amazon_rerank(self):
        """Test initialization with missing API key for Amazon Rerank."""
        with pytest.raises(ValueError):
            config = RerankingConfig(
                model_type="amazon_rerank"
                # Missing api_key
            )
            LLMReranker(config)
    
    def test_init_with_missing_api_key_for_cohere_rerank(self):
        """Test initialization with missing API key for Cohere Rerank."""
        with pytest.raises(ValueError):
            config = RerankingConfig(
                model_type="cohere_rerank"
                # Missing api_key
            )
            LLMReranker(config)
    
    def test_validate_model_parameters_llm_valid(self):
        """Test model parameter validation for LLM with valid parameters."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            model_parameters={
                "temperature": 0.5,
                "max_tokens": 500,
                "top_p": 0.8
            }
        )
        
        # Should not raise an exception
        reranker = LLMReranker(config)
        assert reranker.config.model_parameters["temperature"] == 0.5
    
    def test_validate_model_parameters_llm_invalid_temperature(self):
        """Test model parameter validation for LLM with invalid temperature."""
        with pytest.raises(RerankingConfigError, match="temperature must be between"):
            config = RerankingConfig(
                model_type="llm",
                model_id="us.amazon.nova-lite-v1:0",
                model_parameters={"temperature": 1.5}  # Invalid: > 1.0
            )
            LLMReranker(config)
    
    def test_validate_model_parameters_llm_invalid_max_tokens(self):
        """Test model parameter validation for LLM with invalid max_tokens."""
        with pytest.raises(RerankingConfigError, match="max_tokens must be positive"):
            config = RerankingConfig(
                model_type="llm",
                model_id="us.amazon.nova-lite-v1:0",
                model_parameters={"max_tokens": -100}  # Invalid: negative
            )
            LLMReranker(config)
    
    def test_rerank_candidates_llm(self):
        """Test reranking candidates with generic LLM via Strands."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            top_k_candidates=3
        )
        
        # Mock the Strands agent response
        mock_response = Mock()
        mock_response.message = {
            'content': [{'text': '''
            {
              "rankings": [
                {
                  "class_name": "Technology",
                  "score": 0.95,
                  "reasoning": "High relevance to AI and ML content"
                },
                {
                  "class_name": "Business",
                  "score": 0.3,
                  "reasoning": "Some business relevance"
                },
                {
                  "class_name": "Sports",
                  "score": 0.1,
                  "reasoning": "Low relevance to sports"
                }
              ]
            }
            '''}]
        }
        
        with patch('multi_class_text_classifier.llm_reranker.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = mock_response
            mock_agent_class.return_value = mock_agent
            
            reranker = LLMReranker(config)
            reranked = reranker.rerank_candidates(self.input_text, self.candidates)
            
            # Should return same number of candidates
            assert len(reranked) == len(self.candidates)
            
            # All candidates should have rerank_score set
            for candidate in reranked:
                assert candidate.rerank_score is not None
                assert isinstance(candidate.rerank_score, float)
            
            # Should be sorted by rerank_score (highest first)
            rerank_scores = [c.rerank_score for c in reranked]
            assert rerank_scores == sorted(rerank_scores, reverse=True)
    
    def test_rerank_candidates_amazon_rerank(self):
        """Test reranking candidates with Amazon Rerank."""
        config = RerankingConfig(
            model_type="amazon_rerank",
            api_key="test-api-key",
            top_k_candidates=2
        )
        
        reranker = LLMReranker(config)
        reranked = reranker.rerank_candidates(self.input_text, self.candidates)
        
        # Should return same number of candidates
        assert len(reranked) == len(self.candidates)
        
        # First 2 candidates should have rerank_score set (top_k_candidates=2)
        for i, candidate in enumerate(reranked):
            if i < 2:
                assert candidate.rerank_score is not None
            # Note: In the current implementation, all get rerank scores due to placeholder logic
    
    def test_rerank_candidates_cohere_rerank(self):
        """Test reranking candidates with Cohere Rerank."""
        config = RerankingConfig(
            model_type="cohere_rerank",
            api_key="test-cohere-key",
            top_k_candidates=5
        )
        
        reranker = LLMReranker(config)
        reranked = reranker.rerank_candidates(self.input_text, self.candidates)
        
        # Should return same number of candidates
        assert len(reranked) == len(self.candidates)
        
        # All candidates should have rerank_score set
        for candidate in reranked:
            assert candidate.rerank_score is not None
    
    def test_rerank_candidates_empty_input_text(self):
        """Test reranking with empty input text."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            
            with pytest.raises(RerankingError, match="Input text cannot be empty"):
                reranker.rerank_candidates("", self.candidates)
            
            with pytest.raises(RerankingError, match="Input text cannot be empty"):
                reranker.rerank_candidates("   ", self.candidates)
    
    def test_rerank_candidates_empty_candidates_list(self):
        """Test reranking with empty candidates list."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            result = reranker.rerank_candidates(self.input_text, [])
            
            # Should return empty list
            assert result == []
    
    def test_rerank_candidates_limits_to_top_k(self):
        """Test that reranking limits candidates to top_k_candidates."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            top_k_candidates=2  # Limit to 2 candidates
        )
        
        # Mock the Strands agent response for only 2 candidates
        mock_response = Mock()
        mock_response.message = {
            'content': [{'text': '''
            {
              "rankings": [
                {
                  "class_name": "Technology",
                  "score": 0.95,
                  "reasoning": "High relevance to AI and ML content"
                },
                {
                  "class_name": "Sports",
                  "score": 0.6,
                  "reasoning": "Moderate relevance"
                }
              ]
            }
            '''}]
        }
        
        with patch('multi_class_text_classifier.llm_reranker.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = mock_response
            mock_agent_class.return_value = mock_agent
            
            reranker = LLMReranker(config)
            reranked = reranker.rerank_candidates(self.input_text, self.candidates)
            
            # Should still return all candidates
            assert len(reranked) == len(self.candidates)
            
            # Only the first 2 candidates should have rerank scores (top_k_candidates=2)
            # The remaining candidates should not have rerank scores
            reranked_count = 0
            non_reranked_count = 0
            
            for candidate in reranked:
                if candidate.rerank_score is not None:
                    reranked_count += 1
                    assert isinstance(candidate.rerank_score, float)
                else:
                    non_reranked_count += 1
            
            # Should have exactly 2 reranked candidates and 1 non-reranked
            assert reranked_count == 2
            assert non_reranked_count == 1
    
    def test_fallback_on_error_enabled(self):
        """Test fallback behavior when reranking fails and fallback is enabled."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            fallback_on_error=True
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            reranker = LLMReranker(config)
            
            # Mock the _call_llm method to raise an exception
            with patch.object(reranker, '_call_llm', side_effect=Exception("API Error")):
                result = reranker.rerank_candidates(self.input_text, self.candidates)
                
                # Should return original candidates (fallback)
                assert result == self.candidates
    
    def test_fallback_on_error_disabled(self):
        """Test behavior when reranking fails and fallback is disabled."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            fallback_on_error=False
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            reranker = LLMReranker(config)
            
            # Mock the _call_llm method to raise an exception
            with patch.object(reranker, '_call_llm', side_effect=Exception("API Error")):
                with pytest.raises(RerankingError, match="Reranking failed"):
                    reranker.rerank_candidates(self.input_text, self.candidates)
    
    def test_get_model_parameters_with_defaults(self):
        """Test getting model parameters with defaults applied."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            params = reranker._get_model_parameters("llm")
            
            # Should include default parameters
            assert "temperature" in params
            assert "max_tokens" in params
            assert "top_p" in params
            assert params["temperature"] == 0.1  # Default value
    
    def test_get_model_parameters_with_user_overrides(self):
        """Test getting model parameters with user overrides."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            model_parameters={
                "temperature": 0.7,  # Override default
                "custom_param": "custom_value"  # Additional parameter
            }
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            params = reranker._get_model_parameters("llm")
            
            # Should include user overrides
            assert params["temperature"] == 0.7  # User override
            assert params["custom_param"] == "custom_value"  # User addition
            assert params["max_tokens"] == 1000  # Default value
    
    def test_construct_llm_prompt(self):
        """Test LLM prompt construction."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            prompt = reranker._construct_llm_prompt(self.input_text, self.candidates)
            
            # Should include input text
            assert self.input_text in prompt
            
            # Should include all candidate names and descriptions
            for candidate in self.candidates:
                assert candidate.class_definition.name in prompt
                assert candidate.class_definition.description in prompt
            
            # Should include ranking instructions
            assert "rank" in prompt.lower()
            assert "relevant" in prompt.lower()
            assert "JSON" in prompt  # Should request JSON format
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        config = RerankingConfig(
            model_type="cohere_rerank",
            api_key="secret-key",
            top_k_candidates=7,
            model_parameters={"model": "custom-model"}
        )
        
        reranker = LLMReranker(config)
        summary = reranker.get_config_summary()
        
        # Should include key configuration details
        assert summary["model_type"] == "cohere_rerank"
        assert summary["top_k_candidates"] == 7
        assert summary["fallback_on_error"] == True  # Default value
        assert summary["has_api_key"] == True  # Should not expose actual key
        assert summary["model_parameters"] == {"model": "custom-model"}
    
    def test_validate_connection_valid_config(self):
        """Test connection validation with valid configuration."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            assert reranker.validate_connection() == True
    
    def test_validate_connection_invalid_config(self):
        """Test connection validation with invalid configuration."""
        # Create a reranker with valid config first
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            
            # Then modify config to be invalid and test validation
            reranker.config.model_type = "invalid_model"
            
            # Mock _validate_config to raise an exception
            with patch.object(reranker, '_validate_config', side_effect=RerankingConfigError("Invalid")):
                assert reranker.validate_connection() == False
    
    def test_rerank_candidates_preserves_original_data(self):
        """Test that reranking preserves original candidate data."""
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0"
        )
        
        # Mock the Strands agent response
        mock_response = Mock()
        mock_response.message = {
            'content': [{'text': '''
            {
              "rankings": [
                {
                  "class_name": "Technology",
                  "score": 0.95,
                  "reasoning": "High relevance to AI and ML content"
                },
                {
                  "class_name": "Sports",
                  "score": 0.6,
                  "reasoning": "Moderate relevance"
                },
                {
                  "class_name": "Business",
                  "score": 0.3,
                  "reasoning": "Some business relevance"
                }
              ]
            }
            '''}]
        }
        
        with patch('multi_class_text_classifier.llm_reranker.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.return_value = mock_response
            mock_agent_class.return_value = mock_agent
            
            reranker = LLMReranker(config)
            reranked = reranker.rerank_candidates(self.input_text, self.candidates)
            
            # Should preserve all original data
            for original, reranked_candidate in zip(self.candidates, reranked):
                assert reranked_candidate.class_definition == original.class_definition
                assert reranked_candidate.confidence == original.confidence
                assert reranked_candidate.similarity_score == original.similarity_score
                # Only rerank_score should be added/modified
                assert reranked_candidate.rerank_score is not None
    
    def test_unsupported_model_type_in_rerank_candidates(self):
        """Test handling of unsupported model type during reranking."""
        # Create a valid config first
        config = RerankingConfig(
            model_type="llm",
            model_id="us.amazon.nova-lite-v1:0",
            fallback_on_error=False
        )
        
        with patch('multi_class_text_classifier.llm_reranker.Agent'):
            reranker = LLMReranker(config)
            
            # Modify the model type to something unsupported after initialization
            reranker.config.model_type = "unsupported_model"
            
            with pytest.raises(RerankingError, match="Unsupported model type"):
                reranker.rerank_candidates(self.input_text, self.candidates)