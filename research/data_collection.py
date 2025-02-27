"""
Data collection module for the Personal Research Assistant.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import get_config
from utils.llm_utils import get_llm
from integrations.google_search import search_google
from integrations.wikipedia import search_wikipedia
from integrations.academic_db import search_academic_databases
from integrations.document_ai import process_document

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

# Define prompt templates
CONTENT_EXTRACTION_TEMPLATE = """
You are a research assistant extracting key information from search results.

Search Query: {search_query}
Raw Content:
{raw_content}

Please extract the most relevant information related to the search query.
Focus on factual information, key insights, and important details.
Ignore advertisements, irrelevant sections, and navigation elements.

Your response should be in the following format:
1. A brief summary (2-3 sentences) of the content
2. 3-5 key points or facts extracted from the content
3. Any important dates, statistics, or quotes (with attribution)

Use clear, concise language and focus on accuracy.
"""

def _extract_content(raw_content: str, search_query: str) -> Dict[str, Any]:
    """
    Extract relevant content from raw search results.
    
    Args:
        raw_content: Raw content from search results
        search_query: Original search query
        
    Returns:
        Dict[str, Any]: Extracted content
    """
    # Create prompt
    prompt = PromptTemplate(
        template=CONTENT_EXTRACTION_TEMPLATE,
        input_variables=["search_query", "raw_content"]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            search_query=search_query,
            raw_content=raw_content[:4000]  # Truncate to avoid token limits
        )
        
        return {
            "extracted_content": response,
            "raw_content": raw_content[:8000]  # Store truncated raw content
        }
        
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return {
            "extracted_content": "Error extracting content.",
            "raw_content": raw_content[:1000]  # Store minimal raw content
        }

def _search_and_extract(
    search_function, 
    query: str, 
    topic: Dict[str, Any],
    source_type: str,
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for content and extract relevant information.
    
    Args:
        search_function: Function to use for searching
        query: Search query
        topic: Research topic
        source_type: Type of source
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results with extracted content
    """
    try:
        # Perform search
        search_results = search_function(query, max_results=max_results)
        
        results = []
        for result in search_results:
            # Extract content
            extracted = _extract_content(result["content"], query)
            
            # Create result object
            result_obj = {
                "title": result["title"],
                "url": result["url"],
                "source_type": source_type,
                "query": query,
                "topic": topic["title"],
                "parent_topic": topic.get("parent_topic", ""),
                "extracted_content": extracted["extracted_content"],
                "raw_content": extracted["raw_content"],
                "metadata": result.get("metadata", {}),
                "timestamp": time.time()
            }
            
            results.append(result_obj)
            
        return results
    
    except Exception as e:
        logger.error(f"Error in search and extract: {str(e)}")
        return []

def collect_data(
    topic: Dict[str, Any], 
    include_academic: bool = True, 
    include_news: bool = True,
    max_sources: int = 20
) -> List[Dict[str, Any]]:
    """
    Collect data for a research topic from various sources.
    
    Args:
        topic: Research topic
        include_academic: Whether to include academic sources
        include_news: Whether to include news articles
        max_sources: Maximum number of sources to collect
        
    Returns:
        List[Dict[str, Any]]: Collected data
    """
    logger.info(f"Collecting data for topic: {topic['title']}")
    
    # Determine how many sources to collect from each source type
    source_allocation = {}
    
    if include_academic and include_news:
        source_allocation = {
            "web": int(max_sources * 0.4),
            "wikipedia": int(max_sources * 0.1),
            "academic": int(max_sources * 0.4),
            "news": int(max_sources * 0.1)
        }
    elif include_academic:
        source_allocation = {
            "web": int(max_sources * 0.5),
            "wikipedia": int(max_sources * 0.1),
            "academic": int(max_sources * 0.4),
            "news": 0
        }
    elif include_news:
        source_allocation = {
            "web": int(max_sources * 0.7),
            "wikipedia": int(max_sources * 0.1),
            "academic": 0,
            "news": int(max_sources * 0.2)
        }
    else:
        source_allocation = {
            "web": int(max_sources * 0.8),
            "wikipedia": int(max_sources * 0.2),
            "academic": 0,
            "news": 0
        }
    
    # Ensure we use the search query from the topic
    search_query = topic.get("search_query", f"{topic['title']}")
    
    # Set up search tasks
    search_tasks = []
    
    if source_allocation["web"] > 0:
        search_tasks.append(
            ("web", lambda: _search_and_extract(
                search_google, 
                search_query, 
                topic,
                "web",
                source_allocation["web"]
            ))
        )
    
    if source_allocation["wikipedia"] > 0:
        search_tasks.append(
            ("wikipedia", lambda: _search_and_extract(
                search_wikipedia, 
                topic["title"], 
                topic,
                "wikipedia",
                source_allocation["wikipedia"]
            ))
        )
    
    if source_allocation["academic"] > 0:
        search_tasks.append(
            ("academic", lambda: _search_and_extract(
                search_academic_databases, 
                search_query, 
                topic,
                "academic",
                source_allocation["academic"]
            ))
        )
    
    if source_allocation["news"] > 0:
        # Add "news" to the query for news-specific results
        news_query = f"{search_query} news"
        search_tasks.append(
            ("news", lambda: _search_and_extract(
                search_google, 
                news_query, 
                topic,
                "news",
                source_allocation["news"]
            ))
        )
    
    # Execute search tasks in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(task): source for source, task in search_tasks}
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                results = future.result()
                logger.info(f"Collected {len(results)} results from {source}")
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error collecting data from {source}: {str(e)}")
    
    logger.info(f"Total collected results: {len(all_results)}")
    return all_results
