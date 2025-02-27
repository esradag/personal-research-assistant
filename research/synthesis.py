"""
Content synthesis module for the Personal Research Assistant.
"""
import logging
from typing import List, Dict, Any, Optional, Callable

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llama_index.core import VectorStoreIndex

from config import get_config
from utils.llm_utils import get_llm

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

# Define prompt templates
TOPIC_SYNTHESIS_TEMPLATE = """
You are a research expert synthesizing information on a specific topic.

Topic: {topic}
Main Research Focus: {main_topic}

Below is the information collected from various sources on this topic:

{sources_content}

Please synthesize this information into a coherent, comprehensive understanding of the topic.
Your synthesis should:
1. Identify key insights and findings
2. Resolve contradictions between sources (if any)
3. Highlight areas of consensus and disagreement
4. Identify any gaps in the information

Your response should be in the following JSON format:
```json
{
  "key_findings": ["Finding 1", "Finding 2", ...],
  "consensus_points": ["Point 1", "Point 2", ...],
  "contradictions": ["Contradiction 1", "Contradiction 2", ...],
  "information_gaps": ["Gap 1", "Gap 2", ...],
  "synthesis": "A comprehensive synthesis of the information (500-800 words)",
  "key_sources": ["Source URL 1", "Source URL 2", ...]
}
```

Focus on accuracy, clarity, and objectivity in your synthesis.
"""

OVERALL_SYNTHESIS_TEMPLATE = """
You are a research expert providing a comprehensive synthesis on a complex topic.

Main Research Topic: {main_topic}
Subtopics: {subtopics}

You have access to synthesized information on each subtopic. Your task is to create an overall synthesis
that integrates these individual pieces into a coherent, comprehensive understanding of the main topic.

Here are the individual subtopic syntheses:

{subtopic_syntheses}

Please provide an overall synthesis that:
1. Identifies the main themes and insights across all subtopics
2. Explains how the subtopics relate to and inform each other
3. Highlights any overarching patterns, trends, or principles
4. Identifies remaining research questions or areas for further exploration

Your response should be in the following JSON format:
```json
{
  "main_themes": ["Theme 1", "Theme 2", ...],
  "relationships": ["Relationship 1", "Relationship 2", ...],
  "patterns": ["Pattern 1", "Pattern 2", ...],
  "further_research": ["Question 1", "Question 2", ...],
  "overall_synthesis": "A comprehensive synthesis of all information (800-1200 words)"
}
```

Be thorough, nuanced, and balanced in your synthesis.
"""

def _format_sources_content(sources: List[Dict[str, Any]], topic: str) -> str:
    """
    Format sources content for a specific topic.
    
    Args:
        sources: List of sources
        topic: Topic to filter sources
        
    Returns:
        str: Formatted sources content
    """
    # Filter sources for this topic
    topic_sources = [s for s in sources if s.get("topic", "") == topic]
    
    # Format content
    content_parts = []
    for i, source in enumerate(topic_sources):
        content = source.get("extracted_content", "")
        if not content:
            continue
            
        source_type = source.get("source_type", "unknown")
        url = source.get("url", "unknown")
        reliability = source.get("reliability_score", 0.5)
        
        content_parts.append(
            f"SOURCE {i+1} [{source_type} | Reliability: {reliability:.2f}]:\n"
            f"URL: {url}\n"
            f"{content}\n"
            f"{'='*50}\n"
        )
    
    return "\n".join(content_parts)

def synthesize_topic(
    main_topic: str,
    topic: str,
    sources: List[Dict[str, Any]],
    index: Optional[VectorStoreIndex] = None
) -> Dict[str, Any]:
    """
    Synthesize information for a specific topic.
    
    Args:
        main_topic: Main research topic
        topic: Specific topic to synthesize
        sources: List of sources
        index: Vector store index for querying additional information
        
    Returns:
        Dict[str, Any]: Synthesized information
    """
    logger.info(f"Synthesizing information for topic: {topic}")
    
    # Format sources content
    sources_content = _format_sources_content(sources, topic)
    
    # Create prompt
    prompt = PromptTemplate(
        template=TOPIC_SYNTHESIS_TEMPLATE,
        input_variables=["topic", "main_topic", "sources_content"]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            topic=topic,
            main_topic=main_topic,
            sources_content=sources_content
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
            
        synthesis_results = json.loads(json_str)
        
        # Add topic information
        synthesis_results["topic"] = topic
        
        return synthesis_results
        
    except Exception as e:
        logger.error(f"Error synthesizing topic: {str(e)}")
        # Return a basic structure if synthesis fails
        return {
            "topic": topic,
            "key_findings": ["Information synthesis failed"],
            "consensus_points": [],
            "contradictions": [],
            "information_gaps": ["Complete information unavailable"],
            "synthesis": f"Error synthesizing information for {topic}: {str(e)}",
            "key_sources": []
        }

def synthesize_overall(
    main_topic: str,
    subtopics: List[str],
    topic_syntheses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create an overall synthesis of all topic syntheses.
    
    Args:
        main_topic: Main research topic
        subtopics: List of subtopics
        topic_syntheses: List of topic syntheses
        
    Returns:
        Dict[str, Any]: Overall synthesis
    """
    logger.info(f"Creating overall synthesis for {main_topic}")
    
    # Format subtopic syntheses
    syntheses_text = ""
    for synthesis in topic_syntheses:
        topic = synthesis.get("topic", "Unknown Topic")
        content = synthesis.get("synthesis", "No synthesis available")
        key_findings = synthesis.get("key_findings", [])
        
        syntheses_text += f"TOPIC: {topic}\n\n"
        syntheses_text += "KEY FINDINGS:\n"
        for i, finding in enumerate(key_findings):
            syntheses_text += f"- {finding}\n"
        syntheses_text += "\n"
        syntheses_text += f"SYNTHESIS:\n{content}\n\n"
        syntheses_text += f"{'='*50}\n\n"
    
    # Create prompt
    prompt = PromptTemplate(
        template=OVERALL_SYNTHESIS_TEMPLATE,
        input_variables=["main_topic", "subtopics", "subtopic_syntheses"]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            main_topic=main_topic,
            subtopics=", ".join(subtopics),
            subtopic_syntheses=syntheses_text
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
            
        overall_synthesis = json.loads(json_str)
        
        # Add topic syntheses
        overall_synthesis["topic_syntheses"] = topic_syntheses
        
        return overall_synthesis
        
    except Exception as e:
        logger.error(f"Error creating overall synthesis: {str(e)}")
        # Return a basic structure if synthesis fails
        return {
            "main_themes": ["Overall synthesis failed"],
            "relationships": [],
            "patterns": [],
            "further_research": ["Complete analysis unavailable"],
            "overall_synthesis": f"Error creating overall synthesis: {str(e)}",
            "topic_syntheses": topic_syntheses
        }

def synthesize_content(
    main_topic: str,
    subtopics: List[str],
    verified_data: List[Dict[str, Any]],
    index: Optional[VectorStoreIndex] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[str, Any]:
    """
    Synthesize verified content into a comprehensive understanding.
    
    Args:
        main_topic: Main research topic
        subtopics: List of subtopics
        verified_data: Verified research data
        index: Vector store index for querying additional information
        progress_callback: Callback for progress updates
        
    Returns:
        Dict[str, Any]: Synthesized content
    """
    logger.info(f"Synthesizing content for {len(subtopics)} topics")
    
    # Group sources by topic
    topic_sources = {}
    for source in verified_data:
        topic = source.get("topic", "")
        if topic not in topic_sources:
            topic_sources[topic] = []
        topic_sources[topic].append(source)
    
    # Synthesize each topic
    topic_syntheses = []
    for i, topic in enumerate(subtopics):
        if progress_callback:
            progress = min(0.8, (i + 1) / len(subtopics))
            progress_callback(progress)
            
        # Get sources for this topic
        sources = topic_sources.get(topic, [])
        
        # If no direct sources, use parent topic or related topics
        if not sources:
            for source in verified_data:
                parent_topic = source.get("parent_topic", "")
                if parent_topic == topic:
                    sources.append(source)
        
        # Synthesize topic
        synthesis = synthesize_topic(main_topic, topic, verified_data, index)
        topic_syntheses.append(synthesis)
    
    # Create overall synthesis
    overall_synthesis = synthesize_overall(main_topic, subtopics, topic_syntheses)
    
    # Update progress
    if progress_callback:
        progress_callback(1.0)
    
    return overall_synthesis
