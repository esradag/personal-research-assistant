"""
Main research orchestration module for the Personal Research Assistant.
"""
import logging
import time
from typing import List, Dict, Any, Callable, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from config import get_config
from research.topic_discovery import expand_topic
from research.data_collection import collect_data
from research.verification import verify_information
from research.synthesis import synthesize_content
from research.reporting import generate_report
from utils.llm_utils import get_llm
from utils.langchain_utils import get_memory
from utils.llamaindex_utils import create_index

# Get configuration
config = get_config()

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.logging_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ResearchState(BaseModel):
    """State model for the research flow."""
    research_id: str = Field(description="Unique identifier for the research session")
    main_topic: str = Field(description="Main research topic")
    subtopics: List[Dict[str, Any]] = Field(default_factory=list, description="List of subtopics to research")
    depth_level: str = Field(description="Depth level of the research (Basic, Standard, Comprehensive, Expert)")
    include_academic: bool = Field(default=True, description="Whether to include academic sources")
    include_news: bool = Field(default=True, description="Whether to include news articles")
    expanded_topics: List[Dict[str, Any]] = Field(default_factory=list, description="Expanded research topics")
    collected_data: List[Dict[str, Any]] = Field(default_factory=list, description="Collected research data")
    verified_data: List[Dict[str, Any]] = Field(default_factory=list, description="Verified research data")
    synthesized_content: Dict[str, Any] = Field(default_factory=dict, description="Synthesized research content")
    final_report: Dict[str, Any] = Field(default_factory=dict, description="Final research report")
    current_stage: str = Field(default="init", description="Current research stage")
    progress: float = Field(default=0.0, description="Research progress (0-1)")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during research")

class ResearchEngine:
    """
    Main engine that orchestrates the research process using LangGraph.
    """
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm()
        self.memory = get_memory()
        self.progress_callback = None
        self.graph = self._build_research_graph()
    
    def _build_research_graph(self) -> StateGraph:
        """
        Build the research state graph using LangGraph.
        
        Returns:
            StateGraph: The research flow graph.
        """
        # Create a new state graph
        graph = StateGraph(ResearchState)
        
        # Add nodes to the graph
        graph.add_node("topic_expansion", self._expand_topics)
        graph.add_node("data_collection", self._collect_data)
        graph.add_node("information_verification", self._verify_information)
        graph.add_node("content_synthesis", self._synthesize_content)
        graph.add_node("report_generation", self._generate_report)
        
        # Define edges between nodes
        graph.add_edge("topic_expansion", "data_collection")
        graph.add_edge("data_collection", "information_verification")
        graph.add_edge("information_verification", "content_synthesis")
        graph.add_edge("content_synthesis", "report_generation")
        graph.add_edge("report_generation", END)
        
        # Compile the graph
        return graph.compile()
    
    def _update_progress(self, state: ResearchState, stage: str, progress: float) -> None:
        """
        Update the progress of the research process.
        
        Args:
            state: Current research state
            stage: Current research stage
            progress: Current progress (0-1) 
        """
        state.current_stage = stage
        state.progress = progress
        if self.progress_callback:
            self.progress_callback(stage, progress)
    
    def _expand_topics(self, state: ResearchState) -> ResearchState:
        """
        Expand the research topics.
        
        Args:
            state: Current research state
            
        Returns:
            ResearchState: Updated research state
        """
        try:
            self._update_progress(state, "Expanding Research Topics", 0.1)
            expanded_topics = []
            
            for subtopic in state.subtopics:
                expanded = expand_topic(
                    main_topic=state.main_topic,
                    subtopic=subtopic,
                    depth_level=state.depth_level
                )
                expanded_topics.extend(expanded)
            
            state.expanded_topics = expanded_topics
            self._update_progress(state, "Topic Expansion Completed", 0.2)
            
        except Exception as e:
            logger.error(f"Error in topic expansion: {str(e)}")
            state.errors.append(f"Topic expansion error: {str(e)}")
            
        return state
    
    def _collect_data(self, state: ResearchState) -> ResearchState:
        """
        Collect data for research topics.
        
        Args:
            state: Current research state
            
        Returns:
            ResearchState: Updated research state
        """
        try:
            self._update_progress(state, "Collecting Data", 0.2)
            collected_data = []
            
            for i, topic in enumerate(state.expanded_topics):
                progress = 0.2 + (0.3 * ((i + 1) / len(state.expanded_topics)))
                self._update_progress(state, f"Collecting Data ({i+1}/{len(state.expanded_topics)})", progress)
                
                data = collect_data(
                    topic=topic,
                    include_academic=state.include_academic,
                    include_news=state.include_news,
                    max_sources=self.config.research.max_sources
                )
                collected_data.extend(data)
            
            state.collected_data = collected_data
            self._update_progress(state, "Data Collection Completed", 0.5)
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            state.errors.append(f"Data collection error: {str(e)}")
            
        return state
    
    def _verify_information(self, state: ResearchState) -> ResearchState:
        """
        Verify collected information.
        
        Args:
            state: Current research state
            
        Returns:
            ResearchState: Updated research state
        """
        try:
            self._update_progress(state, "Verifying Information", 0.5)
            
            verified_data = verify_information(
                data=state.collected_data,
                threshold=self.config.research.verification_threshold,
                progress_callback=lambda progress: self._update_progress(
                    state, "Verifying Information", 0.5 + (progress * 0.1)
                )
            )
            
            state.verified_data = verified_data
            self._update_progress(state, "Information Verification Completed", 0.6)
            
        except Exception as e:
            logger.error(f"Error in information verification: {str(e)}")
            state.errors.append(f"Information verification error: {str(e)}")
            
        return state
    
    def _synthesize_content(self, state: ResearchState) -> ResearchState:
        """
        Synthesize verified content.
        
        Args:
            state: Current research state
            
        Returns:
            ResearchState: Updated research state
        """
        try:
            self._update_progress(state, "Synthesizing Content", 0.6)
            
            # Create index from verified data
            index = create_index(state.verified_data)
            
            synthesized_content = synthesize_content(
                main_topic=state.main_topic,
                subtopics=[t["title"] for t in state.subtopics],
                verified_data=state.verified_data,
                index=index,
                progress_callback=lambda progress: self._update_progress(
                    state, "Synthesizing Content", 0.6 + (progress * 0.2)
                )
            )
            
            state.synthesized_content = synthesized_content
            self._update_progress(state, "Content Synthesis Completed", 0.8)
            
        except Exception as e:
            logger.error(f"Error in content synthesis: {str(e)}")
            state.errors.append(f"Content synthesis error: {str(e)}")
            
        return state
    
    def _generate_report(self, state: ResearchState) -> ResearchState:
        """
        Generate the final research report.
        
        Args:
            state: Current research state
            
        Returns:
            ResearchState: Updated research state
        """
        try:
            self._update_progress(state, "Generating Report", 0.8)
            
            final_report = generate_report(
                main_topic=state.main_topic,
                subtopics=state.subtopics,
                synthesized_content=state.synthesized_content,
                sources=state.verified_data,
                citation_style=self.config.research.citation_style,
                max_tokens=self.config.research.max_report_tokens
            )
            
            state.final_report = final_report
            self._update_progress(state, "Report Generation Completed", 1.0)
            
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            state.errors.append(f"Report generation error: {str(e)}")
            
        return state
    
    def run_research_flow(
        self,
        main_topic: str,
        subtopics: List[Dict[str, Any]],
        depth_level: str,
        include_academic: bool,
        include_news: bool,
        research_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete research flow.
        
        Args:
            main_topic: Main research topic
            subtopics: List of subtopics to research
            depth_level: Depth level of the research
            include_academic: Whether to include academic sources
            include_news: Whether to include news articles
            research_id: Unique identifier for the research session
            progress_callback: Callback for progress updates
            
        Returns:
            Dict[str, Any]: Final research report
        """
        self.progress_callback = progress_callback
        
        # Initialize the research state
        initial_state = ResearchState(
            research_id=research_id,
            main_topic=main_topic,
            subtopics=subtopics,
            depth_level=depth_level,
            include_academic=include_academic,
            include_news=include_news
        )
        
        # Run the research flow
        result = self.graph.invoke(initial_state)
        
        # Return the final report
        return result.final_report
