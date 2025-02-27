"""
Information verification module for the Personal Research Assistant.
"""
import logging
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import get_config
from utils.llm_utils import get_llm

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

# Define prompt templates
VERIFICATION_TEMPLATE = """
You are a fact-checking expert assessing the reliability and accuracy of information.

Content to Verify:
{content}

Source Type: {source_type}
Source URL: {url}

Please verify this information by analyzing:
1. Internal consistency: Are there contradictions within the content?
2. Source credibility: Is the source reputable and appropriate for this topic?
3. Factual accuracy: Are the facts consistent with established knowledge?
4. Bias: Is there evidence of bias or slant in the presentation?
5. Completeness: Is the information comprehensive or missing important context?

Rate the information on a scale of 0.0 to 1.0 for each of these dimensions, where:
- 0.0 = Completely unreliable/inaccurate
- 0.5 = Moderately reliable/accurate
- 1.0 = Highly reliable/accurate

Your response should be in the following JSON format:
```json
{
  "consistency_score": 0.0,
  "credibility_score": 0.0,
  "accuracy_score": 0.0,
  "bias_score": 0.0,
  "completeness_score": 0.0,
  "overall_score": 0.0,
  "issues_identified": ["Issue 1", "Issue 2"],
  "verification_notes": "Brief explanation of your assessment"
}
```

Be objective and thorough in your assessment.
"""

CROSS_VERIFICATION_TEMPLATE = """
You are a research expert cross-verifying information across multiple sources.

Topic: {topic}

Source 1:
{source1_content}
Source 1 URL: {source1_url}
Source 1 Type: {source1_type}
Source 1 Reliability Score: {source1_score}

Source 2:
{source2_content}
Source 2 URL: {source2_url}
Source 2 Type: {source2_type}
Source 2 Reliability Score: {source2_score}

Please compare these sources to determine:
1. Areas of agreement between the sources
2. Areas of disagreement or contradiction
3. Information unique to each source
4. Which source appears more reliable or comprehensive on this topic

Your response should be in the following JSON format:
```json
{
  "agreements": ["Area of agreement 1", "Area of agreement 2"],
  "disagreements": ["Disagreement 1", "Disagreement 2"],
  "unique_source1": ["Unique info 1", "Unique info 2"],
  "unique_source2": ["Unique info 1", "Unique info 2"],
  "more_reliable_source": "source1/source2/both equally reliable",
  "cross_verification_notes": "Brief explanation of your assessment"
}
```
"""

def _verify_single_source(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify information from a single source.
    
    Args:
        source: Source data to verify
        
    Returns:
        Dict[str, Any]: Verified source data with reliability scores
    """
    # Create prompt
    prompt = PromptTemplate(
        template=VERIFICATION_TEMPLATE,
        input_variables=["content", "source_type", "url"]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        content = source.get("extracted_content", "")
        if not content:
            content = source.get("raw_content", "")[:2000]  # Use truncated raw content if no extracted content
            
        response = chain.run(
            content=content,
            source_type=source.get("source_type", "unknown"),
            url=source.get("url", "unknown")
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
            
        verification_results = json.loads(json_str)
        
        # Update source with verification results
        source.update(verification_results)
        
        # Set reliability score
        source["reliability_score"] = verification_results.get("overall_score", 0.5)
        
        return source
        
    except Exception as e:
        logger.error(f"Error verifying source: {str(e)}")
        # Set default values if verification fails
        source.update({
            "consistency_score": 0.5,
            "credibility_score": 0.5,
            "accuracy_score": 0.5,
            "bias_score": 0.5,
            "completeness_score": 0.5,
            "overall_score": 0.5,
            "reliability_score": 0.5,
            "issues_identified": ["Verification process failed"],
            "verification_notes": f"Error during verification: {str(e)}"
        })
        return source

def _cross_verify_sources(sources: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    """
    Cross-verify information between pairs of sources on the same topic.
    
    Args:
        sources: List of sources to cross-verify
        topic: Topic of the sources
        
    Returns:
        List[Dict[str, Any]]: Updated sources with cross-verification results
    """
    if len(sources) < 2:
        return sources
    
    # Group sources by topic
    topic_sources = {}
    for source in sources:
        source_topic = source.get("topic", "")
        if source_topic not in topic_sources:
            topic_sources[source_topic] = []
        topic_sources[source_topic].append(source)
    
    # Create pairs of sources for cross-verification
    cross_verifications = []
    for topic_name, topic_sources_list in topic_sources.items():
        if len(topic_sources_list) < 2:
            continue
            
        # Sort sources by reliability score (descending)
        sorted_sources = sorted(
            topic_sources_list, 
            key=lambda x: x.get("reliability_score", 0.0), 
            reverse=True
        )
        
        # Take the top 2 sources for cross-verification
        source1 = sorted_sources[0]
        source2 = sorted_sources[1]
        
        cross_verifications.append((source1, source2, topic_name))
    
    # Create prompt
    prompt = PromptTemplate(
        template=CROSS_VERIFICATION_TEMPLATE,
        input_variables=[
            "topic", 
            "source1_content", "source1_url", "source1_type", "source1_score",
            "source2_content", "source2_url", "source2_type", "source2_score"
        ]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Perform cross-verification
    for source1, source2, topic_name in cross_verifications:
        try:
            response = chain.run(
                topic=topic_name,
                source1_content=source1.get("extracted_content", "")[:1000],
                source1_url=source1.get("url", "unknown"),
                source1_type=source1.get("source_type", "unknown"),
                source1_score=source1.get("reliability_score", 0.5),
                source2_content=source2.get("extracted_content", "")[:1000],
                source2_url=source2.get("url", "unknown"),
                source2_type=source2.get("source_type", "unknown"),
                source2_score=source2.get("reliability_score", 0.5)
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
                
            cross_results = json.loads(json_str)
            
            # Update sources with cross-verification results
            source1["cross_verification"] = cross_results
            source2["cross_verification"] = cross_results
            
            # Adjust reliability scores based on cross-verification
            more_reliable = cross_results.get("more_reliable_source", "both equally reliable")
            
            if more_reliable == "source1":
                # Boost source1, slightly reduce source2
                source1["reliability_score"] = min(1.0, source1.get("reliability_score", 0.5) * 1.1)
                source2["reliability_score"] = max(0.1, source2.get("reliability_score", 0.5) * 0.9)
            elif more_reliable == "source2":
                # Boost source2, slightly reduce source1
                source2["reliability_score"] = min(1.0, source2.get("reliability_score", 0.5) * 1.1)
                source1["reliability_score"] = max(0.1, source1.get("reliability_score", 0.5) * 0.9)
            
        except Exception as e:
            logger.error(f"Error in cross-verification: {str(e)}")
    
    return sources

def verify_information(
    data: List[Dict[str, Any]], 
    threshold: float = 0.7,
    progress_callback: Optional[Callable[[float], None]] = None
) -> List[Dict[str, Any]]:
    """
    Verify collected information from various sources.
    
    Args:
        data: Collected data to verify
        threshold: Reliability threshold for filtering sources
        progress_callback: Callback for progress updates
        
    Returns:
        List[Dict[str, Any]]: Verified data
    """
    logger.info(f"Verifying information from {len(data)} sources")
    
    # Single-source verification
    verified_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(_verify_single_source, source): i for i, source in enumerate(data)}
        for i, future in enumerate(as_completed(future_to_source)):
            try:
                verified_source = future.result()
                verified_data.append(verified_source)
                
                # Update progress
                if progress_callback:
                    progress = min(0.7, (i + 1) / len(data))
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Error in verification process: {str(e)}")
    
    # Cross-verification between sources
    verified_data = _cross_verify_sources(verified_data, "")
    
    # Filter out sources below reliability threshold
    filtered_data = [source for source in verified_data if source.get("reliability_score", 0.0) >= threshold]
    
    # Update progress
    if progress_callback:
        progress_callback(1.0)
    
    logger.info(f"Filtered to {len(filtered_data)} reliable sources")
    return filtered_data
