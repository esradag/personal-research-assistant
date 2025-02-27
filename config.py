"""
Configuration settings for the Personal Research Assistant.
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, Optional, Union

# Load environment variables
load_dotenv()

class OpenAIConfig(BaseModel):
    """OpenAI configuration settings."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-turbo"
    temperature: float = 0.2
    max_tokens: int = 1024

class GoogleConfig(BaseModel):
    """Google Cloud and Search configuration settings."""
    credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    search_api_key: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    search_engine_id: str = os.getenv("GOOGLE_CSE_ID", "")
    storage_bucket: str = os.getenv("GOOGLE_STORAGE_BUCKET", "research-assistant-storage")
    document_ai_processor_id: str = os.getenv("DOCUMENT_AI_PROCESSOR_ID", "")
    document_ai_location: str = os.getenv("DOCUMENT_AI_LOCATION", "us-central1")

class LlamaIndexConfig(BaseModel):
    """LlamaIndex configuration settings."""
    index_path: str = "data/indexes"
    chunk_size: int = 1024
    chunk_overlap: int = 20

class ResearchConfig(BaseModel):
    """Research process configuration settings."""
    max_sources: int = 20
    max_depth: int = 2
    verification_threshold: float = 0.7
    academic_sources_ratio: float = 0.3
    citation_style: str = "APA"
    max_report_tokens: int = 4000

class AppConfig(BaseModel):
    """Main application configuration."""
    openai: OpenAIConfig = OpenAIConfig()
    google: GoogleConfig = GoogleConfig()
    llamaindex: LlamaIndexConfig = LlamaIndexConfig()
    research: ResearchConfig = ResearchConfig()
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    reports_dir: str = "data/reports"
    logging_level: str = "INFO"

# Create the configuration instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the application configuration.
    
    Returns:
        AppConfig: The application configuration.
    """
    return config
