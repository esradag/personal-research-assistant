"""
Academic database integrations for the Personal Research Assistant.
"""
import logging
import time
import requests
from typing import List, Dict, Any

from scholarly import scholarly

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def search_academic_databases(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search academic databases for the given query.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    logger.info(f"Searching academic databases for: {query}")
    
    # Initialize results
    results = []
    
    # Distribute results between databases
    google_scholar_results = max(1, max_results // 2)  # At least 1 result from Google Scholar
    arxiv_results = max(1, max_results - google_scholar_results)  # Rest from arXiv
    
    # Search Google Scholar
    scholar_results = search_google_scholar(query, google_scholar_results)
    results.extend(scholar_results)
    
    # Search arXiv
    arxiv_results = search_arxiv(query, arxiv_results)
    results.extend(arxiv_results)
    
    logger.info(f"Found {len(results)} results from academic databases")
    return results

def search_google_scholar(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search Google Scholar for the given query.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    logger.info(f"Searching Google Scholar for: {query}")
    
    results = []
    
    try:
        # Search Google Scholar
        search_query = scholarly.search_pubs(query)
        
        # Get results
        for _ in range(max_results):
            try:
                # Get next result
                pub = next(search_query)
                
                # Extract author names
                authors = pub.get("bib", {}).get("author", "Unknown")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                
                # Extract publication year
                year = pub.get("bib", {}).get("pub_year", "N/A")
                
                # Extract journal name
                journal = pub.get("bib", {}).get("venue", "Unknown Journal")
                
                # Build content string
                content = f"Title: {pub.get('bib', {}).get('title', 'Unknown Title')}\n"
                content += f"Authors: {authors}\n"
                content += f"Year: {year}\n"
                content += f"Journal/Venue: {journal}\n"
                content += f"Abstract: {pub.get('bib', {}).get('abstract', 'No abstract available')}\n"
                
                # Create result object
                result = {
                    "title": pub.get("bib", {}).get("title", "Unknown Title"),
                    "url": pub.get("pub_url", ""),
                    "content": content,
                    "metadata": {
                        "source": "google_scholar",
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "num_citations": pub.get("num_citations", 0)
                    }
                }
                
                results.append(result)
                
                # Sleep to avoid rate limiting
                time.sleep(1)
                
            except StopIteration:
                break
                
            except Exception as e:
                logger.error(f"Error processing Google Scholar result: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error searching Google Scholar: {str(e)}")
    
    logger.info(f"Found {len(results)} results from Google Scholar")
    return results

def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search arXiv for the given query.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    logger.info(f"Searching arXiv for: {query}")
    
    results = []
    
    try:
        # Format query for arXiv API
        formatted_query = query.replace(" ", "+")
        
        # Set up request
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{formatted_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # Make request
        response = requests.get(base_url, params=params)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"arXiv API error: {response.status_code}")
            return results
        
        # Parse XML response
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, "xml")
        
        # Process entries
        entries = soup.find_all("entry")
        for entry in entries:
            try:
                # Extract entry details
                title = entry.title.text.strip()
                url = entry.id.text.strip()
                summary = entry.summary.text.strip()
                
                # Extract authors
                author_elements = entry.find_all("author")
                authors = ", ".join([author.name.text.strip() for author in author_elements])
                
                # Extract publication date
                published = entry.published.text.strip() if entry.published else "N/A"
                year = published.split("-")[0] if "-" in published else "N/A"
                
                # Extract categories
                category_elements = entry.find_all("category")
                categories = [cat.get("term", "") for cat in category_elements]
                
                # Build content string
                content = f"Title: {title}\n"
                content += f"Authors: {authors}\n"
                content += f"Published: {published}\n"
                content += f"Categories: {', '.join(categories)}\n"
                content += f"Abstract: {summary}\n"
                
                # Create result object
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "metadata": {
                        "source": "arxiv",
                        "authors": authors,
                        "year": year,
                        "categories": categories,
                        "journal": "arXiv"
                    }
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing arXiv entry: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
    
    logger.info(f"Found {len(results)} results from arXiv")
    return results
