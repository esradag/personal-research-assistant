"""
Wikipedia API integration for the Personal Research Assistant.
"""
import logging
import wikipedia
from typing import List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search Wikipedia for the given query.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    logger.info(f"Searching Wikipedia for: {query}")
    
    results = []
    
    try:
        # Search Wikipedia
        search_results = wikipedia.search(query, results=max_results*2)  # Get more results in case some fail
        
        # Process search results
        for title in search_results:
            if len(results) >= max_results:
                break
                
            try:
                # Get page
                page = wikipedia.page(title, auto_suggest=False)
                
                # Create result object
                result = {
                    "title": page.title,
                    "url": page.url,
                    "content": page.content,
                    "metadata": {
                        "source": "wikipedia",
                        "summary": page.summary,
                        "categories": page.categories,
                        "references": page.references,
                        "year": "N/A"  # Wikipedia doesn't provide publication year
                    }
                }
                
                results.append(result)
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first option if disambiguation page
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    
                    # Create result object
                    result = {
                        "title": page.title,
                        "url": page.url,
                        "content": page.content,
                        "metadata": {
                            "source": "wikipedia",
                            "summary": page.summary,
                            "categories": page.categories,
                            "references": page.references,
                            "year": "N/A"
                        }
                    }
                    
                    results.append(result)
                    
                except Exception as inner_e:
                    logger.error(f"Error processing Wikipedia disambiguation: {str(inner_e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing Wikipedia page: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}")
    
    logger.info(f"Found {len(results)} results from Wikipedia")
    return results

def get_wikipedia_summary(topic: str) -> str:
    """
    Get a summary of a Wikipedia topic.
    
    Args:
        topic: Topic to get summary for
        
    Returns:
        str: Summary text
    """
    try:
        # Get summary
        summary = wikipedia.summary(topic, auto_suggest=True)
        return summary
        
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first option if disambiguation page
        try:
            summary = wikipedia.summary(e.options[0], auto_suggest=False)
            return summary
        except Exception as inner_e:
            logger.error(f"Error getting Wikipedia summary for disambiguation: {str(inner_e)}")
            return f"Multiple topics found for '{topic}'. Please specify a more specific topic."
            
    except wikipedia.exceptions.PageError:
        logger.error(f"Wikipedia page not found for: {topic}")
        return f"No Wikipedia page found for '{topic}'."
        
    except Exception as e:
        logger.error(f"Error getting Wikipedia summary: {str(e)}")
        return f"Error retrieving Wikipedia summary for '{topic}'."
