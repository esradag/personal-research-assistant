"""
LlamaIndex helper functions for the Personal Research Assistant.
"""
import logging
import os
from typing import List, Dict, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import OpenAIEmbedding

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Get the embedding model.
    
    Returns:
        OpenAIEmbedding: Embedding model
    """
    return OpenAIEmbedding()

def create_documents_from_data(verified_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Create LlamaIndex documents from verified data.
    
    Args:
        verified_data: List of verified data
        
    Returns:
        List[Document]: List of LlamaIndex documents
    """
    documents = []
    
    for data in verified_data:
        # Extract content
        content = data.get("extracted_content", "")
        if not content:
            content = data.get("raw_content", "")[:2000]  # Use truncated raw content if no extracted content
        
        # Extract metadata
        metadata = {
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "source_type": data.get("source_type", ""),
            "topic": data.get("topic", ""),
            "parent_topic": data.get("parent_topic", ""),
            "reliability_score": data.get("reliability_score", 0.5)
        }
        
        # Create document
        doc = Document(
            text=content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    return documents

def create_index(verified_data: List[Dict[str, Any]], save_path: Optional[str] = None) -> VectorStoreIndex:
    """
    Create a LlamaIndex from verified data.
    
    Args:
        verified_data: List of verified data
        save_path: Path to save the index
        
    Returns:
        VectorStoreIndex: LlamaIndex vector store index
    """
    # Create documents
    documents = create_documents_from_data(verified_data)
    
    # Create node parser
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=config.llamaindex.chunk_size,
        chunk_overlap=config.llamaindex.chunk_overlap
    )
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        embed_model=get_embedding_model()
    )
    
    # Save index if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        index.storage_context.persist(persist_dir=save_path)
    
    return index

def load_index(index_path: str) -> Optional[VectorStoreIndex]:
    """
    Load a LlamaIndex from disk.
    
    Args:
        index_path: Path to load the index from
        
    Returns:
        VectorStoreIndex: LlamaIndex vector store index
    """
    try:
        # Check if index exists
        if not os.path.exists(index_path):
            logger.error(f"Index path does not exist: {index_path}")
            return None
            
        # Load index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context, embed_model=get_embedding_model())
        
        return index
        
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return None

def query_index(index: VectorStoreIndex, query: str, similarity_top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query a LlamaIndex.
    
    Args:
        index: LlamaIndex vector store index
        query: Query string
        similarity_top_k: Number of top results to return
        
    Returns:
        List[Dict[str, Any]]: List of query results
    """
    try:
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k
        )
        
        # Execute query
        response = query_engine.query(query)
        
        # Extract sources
        sources = []
        for source_node in response.source_nodes:
            source = {
                "text": source_node.text,
                "score": source_node.score,
                "metadata": source_node.metadata
            }
            sources.append(source)
        
        return sources
        
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        return []
