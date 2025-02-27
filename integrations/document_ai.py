"""
Google Document AI integration for the Personal Research Assistant.
"""
import logging
import os
from typing import Dict, Any, Optional, BinaryIO

from google.cloud import documentai_v1 as documentai

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def get_document_ai_client():
    """
    Get the Document AI client.
    
    Returns:
        documentai.DocumentProcessorServiceClient: Document AI client
    """
    try:
        # Check for credentials
        if not config.google.credentials_path:
            logger.error("Google Cloud credentials path not found")
            raise ValueError("Google Cloud credentials path not found in configuration")
            
        # Set environment variable for credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.google.credentials_path
        
        # Create client
        client = documentai.DocumentProcessorServiceClient()
        
        return client
        
    except Exception as e:
        logger.error(f"Error creating Document AI client: {str(e)}")
        raise

def get_processor_name():
    """
    Get the Document AI processor name.
    
    Returns:
        str: Processor name
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("Google Cloud project ID not found")
        raise ValueError("Google Cloud project ID not found in environment variables")
        
    location = config.google.document_ai_location
    processor_id = config.google.document_ai_processor_id
    
    if not processor_id:
        logger.error("Document AI processor ID not found")
        raise ValueError("Document AI processor ID not found in configuration")
        
    return f"projects/{project_id}/locations/{location}/processors/{processor_id}"

def process_document(file_content: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Process a document using Google Document AI.
    
    Args:
        file_content: Document content as bytes
        mime_type: MIME type of the document
        
    Returns:
        Dict[str, Any]: Processed document data
    """
    logger.info("Processing document with Document AI")
    
    try:
        # Get client and processor name
        client = get_document_ai_client()
        processor_name = get_processor_name()
        
        # Create document object
        document = {"content": file_content, "mime_type": mime_type}
        
        # Process document
        request = {"name": processor_name, "raw_document": document}
        response = client.process_document(request=request)
        
        # Extract text and entities
        processed_document = response.document
        text = processed_document.text
        
        # Process entities
        entities = []
        for entity in processed_document.entities:
            entities.append({
                "type": entity.type_,
                "mention_text": entity.mention_text,
                "confidence": entity.confidence,
                "page_number": entity.page_anchor.page_refs[0].page if entity.page_anchor.page_refs else 1
            })
        
        # Return processed data
        return {
            "text": text,
            "entities": entities,
            "pages": len(processed_document.pages),
            "mime_type": mime_type
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Return basic data if processing fails
        return {
            "text": "Error processing document",
            "entities": [],
            "pages": 0,
            "mime_type": mime_type,
            "error": str(e)
        }

def process_document_file(file_path: str) -> Dict[str, Any]:
    """
    Process a document file using Google Document AI.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dict[str, Any]: Processed document data
    """
    logger.info(f"Processing document file: {file_path}")
    
    try:
        # Determine MIME type
        mime_type = _get_mime_type(file_path)
        
        # Read file
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        # Process document
        return process_document(file_content, mime_type)
        
    except Exception as e:
        logger.error(f"Error processing document file: {str(e)}")
        return {
            "text": f"Error processing document file: {str(e)}",
            "entities": [],
            "pages": 0,
            "mime_type": "unknown",
            "error": str(e)
        }

def process_pdf_to_text(file: BinaryIO) -> str:
    """
    Process a PDF file to extract text.
    
    Args:
        file: PDF file object
        
    Returns:
        str: Extracted text
    """
    logger.info("Processing PDF to extract text")
    
    try:
        # Read file content
        file_content = file.read()
        
        # Process document
        result = process_document(file_content, "application/pdf")
        
        return result["text"]
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def _get_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MIME type
    """
    import mimetypes
    
    # Initialize mimetypes
    mimetypes.init()
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Default to binary if not found
    if not mime_type:
        mime_type = "application/octet-stream"
        
    return mime_type
