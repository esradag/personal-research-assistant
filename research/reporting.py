"""
Report generation module for the Personal Research Assistant.
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
REPORT_TEMPLATE = """
You are a professional research report writer creating a comprehensive report on a research topic.

Main Research Topic: {main_topic}
Sections to include:
{sections}

For each section, I will provide you with synthesized information. 
Your task is to craft a well-structured, coherent report that presents this information clearly.

The report should:
1. Begin with an executive summary that outlines the key findings
2. Have a logical flow between sections with smooth transitions
3. Use consistent terminology throughout
4. Maintain an objective, authoritative tone
5. Include proper citations using the {citation_style} style
6. End with a conclusion that ties together the main insights

Please format the report with clear headings and subheadings.
Keep the total length around {max_tokens} words.

Here is the synthesized information for each section:

{synthesized_content}

Sources to cite:
{sources}

Your response should be in the following JSON format:
```json
{
  "title": "Research Report Title",
  "summary": "Executive summary (150-250 words)",
  "sections": [
    {
      "title": "Section Title",
      "content": "Section content with appropriate citations"
    },
    ...
  ],
  "conclusion": "Conclusion text",
  "references": ["Reference 1 in {citation_style} format", "Reference 2 in {citation_style} format", ...]
}
```

Remember to maintain academic integrity and properly cite all sources.
"""

def _format_sections(subtopics: List[Dict[str, Any]]) -> str:
    """
    Format sections for the report.
    
    Args:
        subtopics: List of subtopics
        
    Returns:
        str: Formatted sections
    """
    sections = []
    for i, topic in enumerate(subtopics):
        sections.append(f"{i+1}. {topic['title']}")
    
    return "\n".join(sections)

def _format_synthesized_content(synthesized_content: Dict[str, Any]) -> str:
    """
    Format synthesized content for the report.
    
    Args:
        synthesized_content: Synthesized content
        
    Returns:
        str: Formatted synthesized content
    """
    content_parts = []
    
    # Add overall synthesis
    overall = synthesized_content.get("overall_synthesis", "")
    main_themes = synthesized_content.get("main_themes", [])
    
    content_parts.append("OVERALL SYNTHESIS:\n")
    content_parts.append("Main Themes:")
    for theme in main_themes:
        content_parts.append(f"- {theme}")
    content_parts.append("\n")
    content_parts.append(overall)
    content_parts.append("\n" + "="*50 + "\n")
    
    # Add topic syntheses
    topic_syntheses = synthesized_content.get("topic_syntheses", [])
    for synthesis in topic_syntheses:
        topic = synthesis.get("topic", "Unknown Topic")
        content = synthesis.get("synthesis", "No synthesis available")
        key_findings = synthesis.get("key_findings", [])
        
        content_parts.append(f"TOPIC: {topic}\n")
        content_parts.append("Key Findings:")
        for finding in key_findings:
            content_parts.append(f"- {finding}")
        content_parts.append("\n")
        content_parts.append(content)
        content_parts.append("\n" + "="*50 + "\n")
    
    return "\n".join(content_parts)

def _format_sources(sources: List[Dict[str, Any]], citation_style: str) -> str:
    """
    Format sources for the report.
    
    Args:
        sources: List of sources
        citation_style: Citation style
        
    Returns:
        str: Formatted sources
    """
    source_parts = []
    
    for i, source in enumerate(sources):
        title = source.get("title", "Unknown Title")
        url = source.get("url", "")
        source_type = source.get("source_type", "web")
        reliability = source.get("reliability_score", 0.5)
        
        if reliability < 0.6:
            continue  # Skip less reliable sources
            
        if source_type == "academic":
            # Format academic source
            authors = source.get("metadata", {}).get("authors", "Unknown Authors")
            year = source.get("metadata", {}).get("year", "n.d.")
            journal = source.get("metadata", {}).get("journal", "Unknown Journal")
            
            source_parts.append(
                f"[{i+1}] {authors} ({year}). {title}. {journal}. {url}"
            )
        elif source_type == "wikipedia":
            # Format Wikipedia source
            source_parts.append(
                f"[{i+1}] Wikipedia. ({source.get('metadata', {}).get('year', 'n.d.')}). {title}. Retrieved from {url}"
            )
        else:
            # Format web source
            author = source.get("metadata", {}).get("author", "Unknown Author")
            date = source.get("metadata", {}).get("date", "n.d.")
            
            source_parts.append(
                f"[{i+1}] {author} ({date}). {title}. Retrieved from {url}"
            )
    
    return "\n".join(source_parts)

def _format_citations(sources: List[Dict[str, Any]], citation_style: str) -> List[str]:
    """
    Format citations for the report.
    
    Args:
        sources: List of sources
        citation_style: Citation style
        
    Returns:
        List[str]: Formatted citations
    """
    citations = []
    
    for source in sources:
        title = source.get("title", "Unknown Title")
        url = source.get("url", "")
        source_type = source.get("source_type", "web")
        reliability = source.get("reliability_score", 0.5)
        
        if reliability < 0.6:
            continue  # Skip less reliable sources
            
        if source_type == "academic":
            # Format academic citation
            authors = source.get("metadata", {}).get("authors", "Unknown Authors")
            year = source.get("metadata", {}).get("year", "n.d.")
            journal = source.get("metadata", {}).get("journal", "Unknown Journal")
            volume = source.get("metadata", {}).get("volume", "")
            pages = source.get("metadata", {}).get("pages", "")
            
            if citation_style == "APA":
                citation = f"{authors} ({year}). {title}. {journal}"
                if volume:
                    citation += f", {volume}"
                if pages:
                    citation += f", {pages}"
                citation += f". {url}"
            else:  # Default to MLA
                citation = f"{authors}. \"{title}.\" {journal}"
                if volume:
                    citation += f", vol. {volume}"
                if pages:
                    citation += f", {pages}"
                citation += f", {year}. {url}"
                
            citations.append(citation)
            
        elif source_type == "wikipedia":
            # Format Wikipedia citation
            if citation_style == "APA":
                citations.append(
                    f"Wikipedia. ({source.get('metadata', {}).get('year', 'n.d.')}). {title}. Retrieved from {url}"
                )
            else:  # Default to MLA
                citations.append(
                    f"\"{title}.\" Wikipedia, {source.get('metadata', {}).get('year', 'n.d.')}. {url}"
                )
                
        else:
            # Format web citation
            author = source.get("metadata", {}).get("author", "Unknown Author")
            date = source.get("metadata", {}).get("date", "n.d.")
            site = source.get("metadata", {}).get("site", "")
            
            if citation_style == "APA":
                citations.append(
                    f"{author} ({date}). {title}. {site}. {url}"
                )
            else:  # Default to MLA
                citations.append(
                    f"{author}. \"{title}.\" {site}, {date}. {url}"
                )
    
    return citations

def generate_report(
    main_topic: str,
    subtopics: List[Dict[str, Any]],
    synthesized_content: Dict[str, Any],
    sources: List[Dict[str, Any]],
    citation_style: str = "APA",
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Generate a final research report.
    
    Args:
        main_topic: Main research topic
        subtopics: List of subtopics
        synthesized_content: Synthesized content
        sources: List of sources
        citation_style: Citation style
        max_tokens: Maximum tokens for the report
        
    Returns:
        Dict[str, Any]: Final research report
    """
    logger.info(f"Generating report for: {main_topic}")
    
    # Format sections, content, and sources
    sections = _format_sections(subtopics)
    content = _format_synthesized_content(synthesized_content)
    sources_text = _format_sources(sources, citation_style)
    
    # Create prompt
    prompt = PromptTemplate(
        template=REPORT_TEMPLATE,
        input_variables=[
            "main_topic", "sections", "synthesized_content", 
            "sources", "citation_style", "max_tokens"
        ]
    )
    
    # Create chain
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    try:
        response = chain.run(
            main_topic=main_topic,
            sections=sections,
            synthesized_content=content,
            sources=sources_text,
            citation_style=citation_style,
            max_tokens=max_tokens
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
            
        report = json.loads(json_str)
        
        # Format references
        if "references" not in report or not report["references"]:
            report["references"] = _format_citations(sources, citation_style)
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        # Return a basic report if generation fails
        sections_list = [{"title": topic["title"], "content": "Content unavailable"} for topic in subtopics]
        references = _format_citations(sources, citation_style)
        
        return {
            "title": f"Research Report: {main_topic}",
            "summary": f"This report provides an overview of {main_topic}. However, there was an error in the report generation process.",
            "sections": sections_list,
            "conclusion": "Due to technical issues, a complete conclusion could not be generated.",
            "references": references
        }
