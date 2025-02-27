"""
Google Search API integration for the Personal Research Assistant.
"""
import logging
import requests
import time
from typing import List, Dict, Any

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def search_google(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Google using the Custom Search API.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    api_key = config.google.search_api_key
    search_engine_id = config.google.search_engine_id
    
    if not api_key or not search_engine_id:
        logger.error("Google Search API key or Search Engine ID not found")
        return [{"error": "API configuration missing"}]
    
    logger.info(f"Searching Google for: {query}")
    
    # Set up request parameters
    base_url = "https://www.googleapis.com/customsearch/v1"
    
    results = []
    start_index = 1
    
    while len(results) < max_results:
        # Number of results to request in this batch (max 10 per API call)
        num = min(10, max_results - len(results))
        
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": num,
            "start": start_index
        }
        
        try:
            # Make request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Check for errors
            if "error" in data:
                error_msg = data["error"]["message"]
                logger.error(f"Google Search API error: {error_msg}")
                break
                
            # No search results
            if "items" not in data:
                logger.info(f"No more search results for: {query}")
                break
                
            # Process results
            for item in data["items"]:
                # Extract title and link
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                
                # Skip if title or link is missing
                if not title or not link:
                    continue
                
                # Get page content
                content = _fetch_page_content(link)
                
                # Create result object
                result = {
                    "title": title,
                    "url": link,
                    "content": content if content else snippet,
                    "metadata": {
                        "source": "google_search",
                        "date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", ""),
                        "author": item.get("pagemap", {}).get("metatags", [{}])[0].get("author", ""),
                        "site": item.get("displayLink", "")
                    }
                }
                
                results.append(result)
                
                # Break if we have enough results
                if len(results) >= max_results:
                    break
            
            # Update start index for next batch
            start_index += num
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}")
            break
    
    logger.info(f"Found {len(results)} results from Google Search")
    return results

def _fetch_page_content(url: str) -> str:
    """
    Fetch content from a web page.
    
    Args:
        url: URL to fetch
        
    Returns:
        str: Page content
    """
    try:
        # Make request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check response
        if response.status_code != 200:
            logger.warning(f"Failed to fetch page: {url}, status code: {response.status_code}")
            return ""
        
        # Parse HTML and extract text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Error fetching page content: {str(e)}")
        return ""
