"""
OpenAI/LLM utility functions for the Personal Research Assistant.
"""
import logging
import os
from typing import Dict, Any, Optional

from openai import OpenAI
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chat_models import ChatOpenAI

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    """
    Get the OpenAI client instance.
    
    Returns:
        OpenAI: OpenAI client
    """
    api_key = config.openai.api_key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in .env file or config.py")
    
    return OpenAI(api_key=api_key)

def get_llm(model: Optional[str] = None, temperature: Optional[float] = None) -> ChatOpenAI:
    """
    Get a LangChain LLM instance.
    
    Args:
        model: Model name to use
        temperature: Temperature to use
        
    Returns:
        ChatOpenAI: LangChain ChatOpenAI instance
    """
    api_key = config.openai.api_key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in .env file or config.py")
    
    model = model or config.openai.model
    temperature = temperature if temperature is not None else config.openai.temperature
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )

def generate_completion(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate a completion using OpenAI.
    
    Args:
        prompt: Prompt text
        model: Model name to use
        temperature: Temperature to use
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated completion
    """
    client = get_openai_client()
    
    model = model or config.openai.model
    temperature = temperature if temperature is not None else config.openai.temperature
    max_tokens = max_tokens or config.openai.max_tokens
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        raise

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text that might contain markdown code blocks.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Dict[str, Any]: Extracted JSON
    """
    import json
    
    try:
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0]
        else:
            # Assume the whole text is JSON
            json_str = text
            
        # Parse JSON
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"Error extracting JSON from text: {str(e)}")
        raise
