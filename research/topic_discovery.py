"""
Topic discovery and expansion module for the Personal Research Assistant.
"""
import logging
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import get_config
from utils.llm_utils import get_llm

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

# Define prompt templates
TOPIC_DISCOVERY_TEMPLATE = """
You are a research expert helping to identify key aspects of a research topic.

Research Topic: {main_topic}
Research Depth: {depth_level}

Please identify {num_topics} key aspects or subtopics that should be investigated to comprehensively understand this topic.
For each aspect, provide:
1. A clear, concise title (3-7 words)
2. A brief description of why this aspect is important (1-2 sentences)
3. 2-3 key questions that should be answered about this aspect

Your response should be in the following JSON format:
```json
[
  {
    "title": "Aspect Title",
    "description": "Brief description of importance",
    "questions": ["Question 1?", "Question 2?", "Question 3?"]
  },
  ...
]
```

The aspects should be complementary and cover different dimensions of the topic without excessive overlap.
"""

TOPIC_EXPANSION_TEMPLATE = """
You are a research expert expanding on a specific aspect of a broader research topic.

Main Research Topic: {main_topic}
Specific Aspect: {subtopic_title}
Description: {subtopic_description}
Key Questions: {subtopic_questions}
Research Depth: {depth_level}

Based on the research depth ({depth_level}), please expand this aspect into {num_expansions} more specific research points 
that should be investigated. These will guide data collection efforts.

For each expanded point, provide:
1. A focused title (4-8 words)
2. A clear search query that would help find information on this point (15-25 words)
3. 1-2 specific sources types that would be valuable (e.g., "academic papers on machine learning", "industry reports", "government data")

Your response should be in the following JSON format:
```json
[
  {
    "title": "Expanded Point Title",
    "search_query": "Specific search query to find information on this point",
    "source_types": ["Source type 1", "Source type 2"]
  },
  ...
]
```

The expanded points should be detailed enough to guide focused research and data collection.
"""

def _get_num_topics(depth_level: str) -> int:
    """
    Determine the number of topics based on research depth.
    
    Args:
        depth_level: Depth level of the research
        
    Returns:
        int: Number of topics to suggest
    """
    depth_mapping = {
        "Basic": 3,
        "Standard": 5,
        "Comprehensive": 8,
        "Expert": 12
    }
    return depth_mapping.get(depth_level, 5)

def _get_num_expansions(depth_level: str) -> int:
    """
    Determine the number of expansions based on research depth.
    
    Args:
        depth_level: Depth level of the research
        
    Returns:
        int: Number of expansions per topic
    """
    depth_mapping = {
        "Basic": 2,
        "Standard": 3,
        "Comprehensive": 4,
        "Expert": 5
    }
    return depth_mapping.get(depth_level, 3)

def suggest_research_aspects(main_topic: str, depth_level: str) -> List[Dict[str, Any]]:
    """
    Suggest research aspects based on the main topic and depth level.
    
    Args:
        main_topic: Main research topic
        depth_level: Depth level of the research
        
    Returns:
        List[Dict[str, Any]]: List of suggested research aspects
    """
    logger.info(f"Suggesting research aspects for topic: {main_topic}")
    
    num_topics = _get_num_topics(depth_level)
    
    # Create prompt
    prompt = PromptTemplate(
        template=TOPIC_DISCOVERY_TEMPLATE,
        input_variables=["main_topic", "depth_level", "num_topics"]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            main_topic=main_topic,
            depth_level=depth_level,
            num_topics=num_topics
        )
        
        # Parse the JSON response
        import json
        # Extract JSON from the response (might be surrounded by markdown code blocks)
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        else:
            json_str = response
            
        topics = json.loads(json_str)
        
        logger.info(f"Successfully generated {len(topics)} research aspects")
        return topics
        
    except Exception as e:
        logger.error(f"Error suggesting research aspects: {str(e)}")
        # Fallback to a simple structure if there's an error
        return [
            {
                "title": f"{main_topic} - Overview",
                "description": f"General overview of {main_topic}",
                "questions": [f"What is {main_topic}?", f"Why is {main_topic} important?"]
            }
        ]

def expand_topic(main_topic: str, subtopic: Dict[str, Any], depth_level: str) -> List[Dict[str, Any]]:
    """
    Expand a subtopic into more specific research points.
    
    Args:
        main_topic: Main research topic
        subtopic: Subtopic to expand
        depth_level: Depth level of the research
        
    Returns:
        List[Dict[str, Any]]: List of expanded research points
    """
    logger.info(f"Expanding subtopic: {subtopic['title']}")
    
    num_expansions = _get_num_expansions(depth_level)
    
    # Create prompt
    prompt = PromptTemplate(
        template=TOPIC_EXPANSION_TEMPLATE,
        input_variables=[
            "main_topic", "subtopic_title", "subtopic_description", 
            "subtopic_questions", "depth_level", "num_expansions"
        ]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            main_topic=main_topic,
            subtopic_title=subtopic["title"],
            subtopic_description=subtopic["description"],
            subtopic_questions=subtopic["questions"],
            depth_level=depth_level,
            num_expansions=num_expansions
        )
        
        # Parse the JSON response
        import json
        # Extract JSON from the response (might be surrounded by markdown code blocks)
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        else:
            json_str = response
            
        expanded_topics = json.loads(json_str)
        
        # Add parent topic information
        for topic in expanded_topics:
            topic["parent_topic"] = subtopic["title"]
        
        logger.info(f"Successfully expanded subtopic into {len(expanded_topics)} research points")
        return expanded_topics
        
    except Exception as e:
        logger.error(f"Error expanding topic: {str(e)}")
        # Fallback to a simple structure if there's an error
        return [
            {
                "title": f"{subtopic['title']} - Detail",
                "search_query": f"{main_topic} {subtopic['title']} research",
                "source_types": ["Web articles", "Research papers"],
                "parent_topic": subtopic["title"]
            }
        ]
