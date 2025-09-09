"""
PDF text and image extraction module for multi-class text classifier.

This module provides configurable PDF extraction capabilities that can:
1. Extract text content from PDFs
2. Extract and describe images using various AI models
3. Perform OCR on images when needed
4. Support multiple model providers (Nova Lite, Nova Pro, Claude Sonnet 4, etc.)
"""

import io
import base64
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from strands import Agent
from .exceptions import ProcessingError, ConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class ImageContent:
    """Represents extracted image content with description."""
    page_number: int
    image_index: int
    description: str
    ocr_text: str = ""
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDFContent:
    """Represents extracted PDF content including text and images."""
    text_content: str
    images: List[ImageContent] = field(default_factory=list)
    page_count: int = 0
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_content(self) -> str:
        """Get combined text and image descriptions."""
        content_parts = []
        
        if self.text_content.strip():
            content_parts.append("TEXT CONTENT:")
            content_parts.append(self.text_content)
        
        if self.images:
            content_parts.append("\nIMAGE DESCRIPTIONS:")
            for img in self.images:
                content_parts.append(f"\nPage {img.page_number}, Image {img.image_index}:")
                content_parts.append(img.description)
                if img.ocr_text.strip():
                    content_parts.append(f"OCR Text: {img.ocr_text}")
        
        return "\n".join(content_parts)


# PDFExtractorConfig is now replaced by the centralized PDFConfig from config.py


class PDFExtractor:
    """
    PDF content extractor with configurable AI model support.
    
    Extracts text and images from PDFs, using AI models to describe
    image content and perform OCR when needed.
    """
    
    def __init__(self, model_id: Optional[str] = None, **kwargs):
        """
        Initialize PDF extractor with configuration.
        
        Args:
            model_id: Optional model ID override
            **kwargs: Additional configuration options (for backward compatibility)
        """
        from .config import config as classifier_config
        self.config = classifier_config.pdf
        self.model_id = model_id or self.config.model_id
        self._agent = None
        
    def _get_bedrock_client(self):
        """Get or create boto3 Bedrock client for AI processing."""
        if self._agent is None:
            try:
                import boto3
                from botocore.config import Config
                
                # Configure boto3 client with timeout and retry settings
                from .config import config as classifier_config
                config = Config(
                    region_name=classifier_config.aws.bedrock_region,
                    retries={
                        'max_attempts': 3,
                        'mode': 'standard'
                    },
                    read_timeout=60,
                    connect_timeout=10
                )
                
                # Create bedrock runtime client
                self._agent = boto3.client('bedrock-runtime', config=config)
                
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize Bedrock client: {e}")
        return self._agent
    
    def extract_from_file(self, pdf_path: Union[str, Path]) -> PDFContent:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFContent with extracted text and image descriptions
            
        Raises:
            ProcessingError: If PDF processing fails
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                return self.extract_from_bytes(file.read(), str(pdf_path))
        except Exception as e:
            raise ProcessingError(f"Failed to read PDF file {pdf_path}: {e}")
    
    def extract_from_bytes(self, pdf_bytes: bytes, source_name: str = "PDF") -> PDFContent:
        """
        Extract content from PDF bytes.
        
        Args:
            pdf_bytes: PDF file content as bytes
            source_name: Name/identifier for the PDF source
            
        Returns:
            PDFContent with extracted text and image descriptions
            
        Raises:
            ProcessingError: If PDF processing fails
        """
        try:
            # Extract text content (always enabled)
            text_content = self._extract_text(pdf_bytes)
            
            # Extract images (always enabled)
            images = self._extract_images(pdf_bytes)
            
            # Get page count
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
            
            return PDFContent(
                text_content=text_content,
                images=images,
                page_count=page_count,
                extraction_metadata={
                    "source": source_name,
                    "config": {
                        "model_id": self.model_id,
                        "extract_text": True,  # Always extract text for now
                        "extract_images": True  # Always extract images for now
                    }
                }
            )
            
        except Exception as e:
            raise ProcessingError(f"Failed to extract PDF content: {e}")
    
    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF."""
        text_parts = []
        
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---")
                    text_parts.append(text)
            doc.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF text extraction failed: {e}, trying PyPDF2")
            
            # Fallback to PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---")
                        text_parts.append(text)
            except Exception as e2:
                logger.error(f"PyPDF2 text extraction also failed: {e2}")
                raise ProcessingError(f"Text extraction failed with both methods: {e}, {e2}")
        
        return "\n".join(text_parts)
    
    def _extract_images(self, pdf_bytes: bytes) -> List[ImageContent]:
        """Extract and describe images from PDF."""
        images = []
        total_images = 0
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(doc)):
                if total_images >= self.config.max_total_images:
                    logger.warning(f"Reached max total images limit: {self.config.max_total_images}")
                    break
                
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                page_images = 0
                for img_index, img in enumerate(image_list):
                    if page_images >= self.config.max_images_per_page:
                        logger.warning(f"Reached max images per page limit on page {page_num + 1}")
                        break
                    
                    if total_images >= self.config.max_total_images:
                        break
                    
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip very small images
                        if pix.width < self.config.min_image_size or pix.height < self.config.min_image_size:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                        
                        pix = None  # Free memory
                        
                        # Process image with AI
                        image_content = self._process_image(
                            pil_image, 
                            page_num + 1, 
                            img_index + 1
                        )
                        
                        if image_content:
                            images.append(image_content)
                            page_images += 1
                            total_images += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index + 1} on page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            raise ProcessingError(f"Image extraction failed: {e}")
        
        return images
    
    def _process_image(self, pil_image: Image.Image, page_num: int, img_index: int) -> Optional[ImageContent]:
        """Process image with AI model to get description and OCR."""
        try:
            # Resize image if needed
            if max(pil_image.size) > self.config.max_image_size:
                pil_image.thumbnail((self.config.max_image_size, self.config.max_image_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=self.config.image_quality)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create prompt for AI model
            prompt = self._create_image_analysis_prompt()
            
            # Get Bedrock client
            bedrock_client = self._get_bedrock_client()
            
            # Analyze image using Bedrock converse API with tool use
            response = self._call_bedrock_converse(bedrock_client, prompt, img_base64)
            
            # Extract structured response
            description = response.get('description', '')
            ocr_text = response.get('ocr_text', '')
            
            return ImageContent(
                page_number=page_num,
                image_index=img_index,
                description=description,
                ocr_text=ocr_text,
                confidence_score=0.8,  # Default confidence
                metadata={
                    "image_size": pil_image.size,
                    "model_used": self.model_id
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process image {img_index} on page {page_num}: {e}")
            return None
    
    def _create_image_analysis_prompt(self) -> str:
        """Create prompt for image analysis."""
        prompt_parts = [
            "Analyze this image and extract information using the provided tool.",
            "1. Extract any text content visible in the image (OCR)",
            "2. Provide a detailed description in French of any non-text objects (handwritten signature, logo, table, etc.)",
            "Focus on extracting any readable text accurately. "
            "If there's no text visible, use 'No text found' for the OCR text.",
            "Use the extract_image_content tool to provide your analysis in a structured format."
        ]
        
        return "\n".join(prompt_parts)
    
    def _call_bedrock_converse(self, bedrock_client, prompt: str, img_base64: str) -> Dict[str, str]:
        """
        Call Bedrock converse API for image analysis using tool use.
        
        Args:
            bedrock_client: Boto3 Bedrock runtime client
            prompt: Text prompt for image analysis
            img_base64: Base64 encoded image data
            
        Returns:
            Dictionary with 'description' and 'ocr_text' keys
            
        Raises:
            ProcessingError: If API call fails
        """
        try:
            # Define the tool for structured output
            tool_config = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "extract_image_content",
                            "description": "Extract and structure image content including OCR text and descriptions",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "ocr_text": {
                                            "type": "string",
                                            "description": "Any readable text found in the image. Use 'NA' if no text is visible."
                                        },
                                        "description": {
                                            "type": "string", 
                                            "description": "Detailed description of non-text elements like handwritten signatures, logos, tables, diagrams, etc. Use 'NA' if none are present. Your answer must be in the language of the ocr text."
                                        }
                                    },
                                    "required": ["ocr_text", "description"]
                                }
                            }
                        }
                    }
                ]
            }
            
            # Prepare the message for converse API
            message = {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    },
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {
                                "bytes": base64.b64decode(img_base64)
                            }
                        }
                    }
                ]
            }
            
            # Get model parameters from config
            model_params = {
                "maxTokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "topP": self.config.top_p
            }
            
            # Call Bedrock converse API with tool configuration
            response = bedrock_client.converse(
                modelId=self.model_id,
                messages=[message],
                inferenceConfig=model_params,
                toolConfig=tool_config,
            )
            
            # Extract tool use response
            if response.get('stopReason') != 'tool_use':
                raise ProcessingError(f"Expected tool use but got stop reason: {response.get('stopReason')}")
            
            logger.debug("Model requested tool use for structured output")
            content = response['output']['message']['content']
            for content_block in content:
                if 'toolUse' in content_block:
                    tool_input = content_block['toolUse']['input']
                    logger.debug(f"Extracted structured content: {tool_input}")
                    return {
                        'description': tool_input.get('description', ''),
                        'ocr_text': tool_input.get('ocr_text', '')
                    }
            
            raise ProcessingError("Tool use response received but no tool input found")
            
        except Exception as e:
            raise ProcessingError(f"Bedrock converse API call failed: {e}")
    



def create_pdf_extractor(model_id: Optional[str] = None, **kwargs) -> PDFExtractor:
    """
    Create a PDF extractor with common configuration.
    
    Args:
        model_id: Optional model ID override
        **kwargs: Additional configuration options (for backward compatibility)
        
    Returns:
        Configured PDFExtractor instance
    """
    return PDFExtractor(model_id=model_id, **kwargs)