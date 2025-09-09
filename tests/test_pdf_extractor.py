"""
Tests for PDF extraction functionality.
"""

import pytest
import io
import base64
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from multi_class_text_classifier.pdf_extractor import (
    PDFExtractor,
    PDFExtractorConfig,
    PDFContent,
    ImageContent,
    create_pdf_extractor
)
from multi_class_text_classifier.exceptions import ConfigurationError, ProcessingError


class TestPDFExtractorConfig:
    """Test PDFExtractorConfig validation and functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PDFExtractorConfig()
        
        assert config.model_id == "us.amazon.nova-lite-v1:0"
        # aws_region is now accessed from the global config, not PDFConfig
        assert config.extract_text is True
        assert config.extract_images is True
        assert config.max_image_size == 1024
        assert config.max_total_images == 50
    
    def test_custom_model_id(self):
        """Test custom model ID configuration."""
        custom_model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        config = PDFExtractorConfig(model_id=custom_model_id)
        
        assert config.model_id == custom_model_id
    
    def test_different_model_ids(self):
        """Test different model ID configurations."""
        test_cases = [
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ]
        
        for model_id in test_cases:
            config = PDFExtractorConfig(model_id=model_id)
            assert config.model_id == model_id


class TestImageContent:
    """Test ImageContent data model."""
    
    def test_image_content_creation(self):
        """Test ImageContent creation with all fields."""
        content = ImageContent(
            page_number=1,
            image_index=2,
            description="A chart showing sales data",
            ocr_text="Q1 Sales: $100K",
            confidence_score=0.85,
            metadata={"size": (800, 600)}
        )
        
        assert content.page_number == 1
        assert content.image_index == 2
        assert content.description == "A chart showing sales data"
        assert content.ocr_text == "Q1 Sales: $100K"
        assert content.confidence_score == 0.85
        assert content.metadata["size"] == (800, 600)
    
    def test_image_content_defaults(self):
        """Test ImageContent with default values."""
        content = ImageContent(
            page_number=1,
            image_index=1,
            description="Test image"
        )
        
        assert content.ocr_text == ""
        assert content.confidence_score == 0.0
        assert content.metadata == {}


class TestPDFContent:
    """Test PDFContent data model."""
    
    def test_pdf_content_creation(self):
        """Test PDFContent creation."""
        images = [
            ImageContent(1, 1, "First image", "Text 1"),
            ImageContent(2, 1, "Second image", "Text 2")
        ]
        
        content = PDFContent(
            text_content="Sample text content",
            images=images,
            page_count=2,
            extraction_metadata={"source": "test.pdf"}
        )
        
        assert content.text_content == "Sample text content"
        assert len(content.images) == 2
        assert content.page_count == 2
        assert content.extraction_metadata["source"] == "test.pdf"
    
    def test_full_content_text_only(self):
        """Test full_content property with text only."""
        content = PDFContent(
            text_content="This is the main text content.",
            images=[],
            page_count=1
        )
        
        full_content = content.full_content
        assert "TEXT CONTENT:" in full_content
        assert "This is the main text content." in full_content
        assert "IMAGE DESCRIPTIONS:" not in full_content
    
    def test_full_content_with_images(self):
        """Test full_content property with images."""
        images = [
            ImageContent(1, 1, "Chart showing data", "Chart Title"),
            ImageContent(2, 1, "Photo of building", "")
        ]
        
        content = PDFContent(
            text_content="Main text",
            images=images,
            page_count=2
        )
        
        full_content = content.full_content
        assert "TEXT CONTENT:" in full_content
        assert "Main text" in full_content
        assert "IMAGE DESCRIPTIONS:" in full_content
        assert "Page 1, Image 1:" in full_content
        assert "Chart showing data" in full_content
        assert "OCR Text: Chart Title" in full_content
        assert "Page 2, Image 1:" in full_content
        assert "Photo of building" in full_content
    
    def test_full_content_images_only(self):
        """Test full_content property with images only."""
        images = [ImageContent(1, 1, "Single image", "Image text")]
        
        content = PDFContent(
            text_content="",
            images=images,
            page_count=1
        )
        
        full_content = content.full_content
        assert "TEXT CONTENT:" not in full_content
        assert "IMAGE DESCRIPTIONS:" in full_content
        assert "Single image" in full_content


class TestPDFExtractor:
    """Test PDFExtractor functionality."""
    
    def test_extractor_initialization(self):
        """Test PDFExtractor initialization."""
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        assert extractor.config == config
        assert extractor._agent is None
    
    @patch('boto3.client')
    def test_get_bedrock_client_creation(self, mock_boto_client):
        """Test Bedrock client creation and caching."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        # First call should create client
        client1 = extractor._get_bedrock_client()
        mock_boto_client.assert_called_once()
        assert client1 == mock_client
        
        # Second call should return cached client
        client2 = extractor._get_bedrock_client()
        assert client2 == mock_client
        assert mock_boto_client.call_count == 1  # Still only called once
    
    @patch('boto3.client')
    def test_get_bedrock_client_configuration_error(self, mock_boto_client):
        """Test Bedrock client creation failure."""
        mock_boto_client.side_effect = Exception("Client creation failed")
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        with pytest.raises(ConfigurationError, match="Failed to initialize Bedrock client"):
            extractor._get_bedrock_client()
    
    def test_extract_from_file_not_found(self):
        """Test extract_from_file with non-existent file."""
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_file("nonexistent.pdf")
    
    @patch('builtins.open')
    @patch.object(PDFExtractor, 'extract_from_bytes')
    @patch('pathlib.Path.exists')
    def test_extract_from_file_success(self, mock_exists, mock_extract_bytes, mock_open):
        """Test successful extract_from_file."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock file reading
        mock_file = Mock()
        mock_file.read.return_value = b"fake pdf bytes"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock extract_from_bytes
        expected_content = PDFContent("test content", [], 1)
        mock_extract_bytes.return_value = expected_content
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        result = extractor.extract_from_file("test.pdf")
        
        mock_open.assert_called_once()
        mock_extract_bytes.assert_called_once_with(b"fake pdf bytes", "test.pdf")
        assert result == expected_content
    
    @patch('multi_class_text_classifier.pdf_extractor.fitz')
    @patch.object(PDFExtractor, '_extract_text')
    @patch.object(PDFExtractor, '_extract_images')
    def test_extract_from_bytes_success(self, mock_extract_images, mock_extract_text, mock_fitz):
        """Test successful extract_from_bytes."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_fitz.open.return_value = mock_doc
        
        # Mock extraction methods
        mock_extract_text.return_value = "Extracted text"
        mock_extract_images.return_value = [
            ImageContent(1, 1, "Test image", "OCR text")
        ]
        
        config = PDFExtractorConfig(
            extract_text=True,
            extract_images=True
        )
        extractor = PDFExtractor(config)
        
        result = extractor.extract_from_bytes(b"fake pdf", "test.pdf")
        
        assert result.text_content == "Extracted text"
        assert len(result.images) == 1
        assert result.page_count == 3
        assert result.extraction_metadata["source"] == "test.pdf"
        
        mock_extract_text.assert_called_once_with(b"fake pdf")
        mock_extract_images.assert_called_once_with(b"fake pdf")
        mock_doc.close.assert_called_once()
    
    @patch('multi_class_text_classifier.pdf_extractor.fitz')
    def test_extract_from_bytes_text_only(self, mock_fitz):
        """Test extract_from_bytes with text extraction only."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_fitz.open.return_value = mock_doc
        
        config = PDFExtractorConfig(
            extract_text=True,
            extract_images=False
        )
        extractor = PDFExtractor(config)
        
        with patch.object(extractor, '_extract_text', return_value="Text only"):
            result = extractor.extract_from_bytes(b"fake pdf", "test.pdf")
        
        assert result.text_content == "Text only"
        assert result.images == []
        assert result.page_count == 1
    
    def test_create_image_analysis_prompt(self):
        """Test image analysis prompt creation."""
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        prompt = extractor._create_image_analysis_prompt()
        
        assert "Analyze this image" in prompt
        assert "extract_image_content tool" in prompt
        assert "extracting any readable text" in prompt
    
    def test_create_image_analysis_prompt_french_description(self):
        """Test image analysis prompt includes French description requirement."""
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        prompt = extractor._create_image_analysis_prompt()
        
        assert "Analyze this image" in prompt
        assert "extract_image_content tool" in prompt
        assert "French" in prompt
    

    
    @patch.object(PDFExtractor, '_get_bedrock_client')
    def test_call_bedrock_converse_tool_use_success(self, mock_get_client):
        """Test successful tool use with Bedrock converse API."""
        # Mock Bedrock client response with tool use
        mock_client = Mock()
        mock_response = {
            'stopReason': 'tool_use',
            'output': {
                'message': {
                    'content': [
                        {
                            'toolUse': {
                                'input': {
                                    'description': 'A chart showing sales data',
                                    'ocr_text': 'Q1: $100K, Q2: $150K'
                                }
                            }
                        }
                    ]
                }
            }
        }
        mock_client.converse.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        # Use valid base64 string for testing
        valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        result = extractor._call_bedrock_converse(mock_client, "test prompt", valid_base64)
        
        assert result['description'] == 'A chart showing sales data'
        assert result['ocr_text'] == 'Q1: $100K, Q2: $150K'
        
        # Verify the tool configuration was passed correctly
        call_args = mock_client.converse.call_args
        assert 'toolConfig' in call_args.kwargs
        tool_config = call_args.kwargs['toolConfig']
        assert 'tools' in tool_config
        assert len(tool_config['tools']) == 1
        assert tool_config['tools'][0]['toolSpec']['name'] == 'extract_image_content'
    
    @patch.object(PDFExtractor, '_get_bedrock_client')
    def test_call_bedrock_converse_wrong_stop_reason(self, mock_get_client):
        """Test error when model doesn't use tool."""
        # Mock Bedrock client response without tool use
        mock_client = Mock()
        mock_response = {
            'stopReason': 'end_turn',
            'output': {
                'message': {
                    'content': [
                        {
                            'text': 'Some regular text response'
                        }
                    ]
                }
            }
        }
        mock_client.converse.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        # Use valid base64 string for testing
        valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with pytest.raises(ProcessingError, match="Expected tool use but got stop reason"):
            extractor._call_bedrock_converse(mock_client, "test prompt", valid_base64)
    
    @patch.object(PDFExtractor, '_get_bedrock_client')
    def test_call_bedrock_converse_api_error(self, mock_get_client):
        """Test API error handling."""
        mock_client = Mock()
        mock_client.converse.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        config = PDFExtractorConfig()
        extractor = PDFExtractor(config)
        
        # Use valid base64 string for testing
        valid_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        with pytest.raises(ProcessingError, match="Bedrock converse API call failed"):
            extractor._call_bedrock_converse(mock_client, "test prompt", valid_base64)


class TestCreatePDFExtractor:
    """Test the create_pdf_extractor convenience function."""
    
    def test_create_with_defaults(self):
        """Test creating extractor with default settings."""
        extractor = create_pdf_extractor()
        
        assert extractor.config.model_id == "us.amazon.nova-lite-v1:0"
        # aws_region is now accessed from the global config, not PDFConfig
        assert extractor.config.extract_text is True
        assert extractor.config.extract_images is True
    
    def test_create_with_custom_settings(self):
        """Test creating extractor with custom settings."""
        extractor = create_pdf_extractor(
            model_id="us.amazon.nova-pro-v1:0",
            extract_text=False,
            max_image_size=512
        )
        
        assert extractor.config.model_id == "us.amazon.nova-pro-v1:0"
        # aws_region is now accessed from the global config, not PDFConfig
        assert extractor.config.extract_text is False
        assert extractor.config.max_image_size == 512
    
    def test_create_custom_model(self):
        """Test creating extractor with custom model."""
        custom_model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        extractor = create_pdf_extractor(
            model_id=custom_model_id
        )
        
        assert extractor.config.model_id == custom_model_id
        # aws_region is now accessed from the global config, not PDFConfig


# Integration test helpers (these would need real PDF files to run)
class TestPDFExtractorIntegration:
    """Integration tests that would require real PDF files and AWS access."""
    
    @pytest.mark.skip(reason="Requires real PDF file and AWS credentials")
    def test_real_pdf_extraction(self):
        """Test with a real PDF file (skipped by default)."""
        extractor = create_pdf_extractor()
        
        # This would work with a real PDF file
        # content = extractor.extract_from_file("sample.pdf")
        # assert content.text_content
        # assert content.page_count > 0
        pass
    
    @pytest.mark.skip(reason="Requires AWS credentials and model access")
    def test_real_image_analysis(self):
        """Test with real image analysis (skipped by default)."""
        # This would test actual AI model integration
        pass