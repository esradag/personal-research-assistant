"""
Data visualization tools for the Personal Research Assistant.
"""
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

def create_knowledge_graph(research_results: Dict[str, Any]) -> plt.Figure:
    """
    Create a knowledge graph visualization from research results.
    
    Args:
        research_results: Research results
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    try:
        # Create graph
        G = nx.Graph()
        
        # Add main topic node
        main_topic = research_results.get("title", "Research Topic")
        G.add_node(main_topic, type="main_topic", size=20)
        
        # Add section nodes and connect to main topic
        sections = research_results.get("sections", [])
        for section in sections:
            section_title = section.get("title", "Section")
            G.add_node(section_title, type="section", size=15)
            G.add_edge(main_topic, section_title, weight=5)
            
            # Extract key concepts from section content
            content = section.get("content", "")
            # Simple approach: extract capitalized phrases as key concepts
            import re
            concepts = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', content)
            # Take top 5 unique concepts
            unique_concepts = list(set(concepts))[:5]
            
            # Add concept nodes and connect to section
            for concept in unique_concepts:
                if len(concept) > 3:  # Filter out short concepts
                    G.add_node(concept, type="concept", size=10)
                    G.add_edge(section_title, concept, weight=2)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get node attributes
        node_types = nx.get_node_attributes(G, 'type')
        node_sizes = nx.get_node_attributes(G, 'size')
        
        # Define colors for different node types
        color_map = {"main_topic": "red", "section": "blue", "concept": "green"}
        node_colors = [color_map.get(node_types.get(node, "concept"), "gray") for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=[node_sizes.get(node, 5) * 20 for node in G.nodes()],
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v].get('weight', 1) for u, v in G.edges()],
            alpha=0.5,
            edge_color="gray"
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family="sans-serif"
        )
        
        # Set title and remove axis
        plt.title("Research Knowledge Graph", fontsize=15)
        plt.axis("off")
        
        return plt.gcf()  # Get current figure
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}")
        # Create a simple fallback figure
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Knowledge graph visualization failed", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis("off")
        return plt.gcf()

def create_source_pie_chart(sources: List[Dict[str, Any]]) -> px.pie:
    """
    Create a pie chart of source types.
    
    Args:
        sources: List of sources
        
    Returns:
        px.pie: Plotly pie chart
    """
    try:
        # Count source types
        source_types = {}
        for source in sources:
            source_type = source.get("source_type", "unknown")
            if source_type not in source_types:
                source_types[source_type] = 0
            source_types[source_type] += 1
        
        # Create DataFrame
        df = pd.DataFrame({
            "Source Type": list(source_types.keys()),
            "Count": list(source_types.values())
        })
        
        # Create pie chart
        fig = px.pie(
            df,
            values="Count",
            names="Source Type",
            title="Source Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Update layout
        fig.update_layout(
            legend_title="Source Types",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating source pie chart: {str(e)}")
        # Create a simple fallback figure
        df = pd.DataFrame({
            "Source Type": ["Error"],
            "Count": [1]
        })
        fig = px.pie(df, values="Count", names="Source Type", title="Source Distribution Error")
        return fig

def create_reliability_histogram(sources: List[Dict[str, Any]]) -> px.histogram:
    """
    Create a histogram of source reliability scores.
    
    Args:
        sources: List of sources
        
    Returns:
        px.histogram: Plotly histogram
    """
    try:
        # Extract reliability scores
        reliability_scores = [source.get("reliability_score", 0.0) for source in sources]
        
        # Create DataFrame
        df = pd.DataFrame({
            "Reliability Score": reliability_scores,
            "Source Type": [source.get("source_type", "unknown") for source in sources]
        })
        
        # Create histogram
        fig = px.histogram(
            df,
            x="Reliability Score",
            color="Source Type",
            nbins=10,
            range_x=[0, 1],
            title="Source Reliability Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Reliability Score",
            yaxis_title="Number of Sources",
            bargap=0.1
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating reliability histogram: {str(e)}")
        # Create a simple fallback figure
        df = pd.DataFrame({
            "Reliability Score": [0.5],
            "Source Type": ["Error"]
        })
        fig = px.histogram(df, x="Reliability Score", title="Reliability Histogram Error")
        return fig
