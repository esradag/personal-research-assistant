"""
Main Streamlit application for the Personal Research Assistant.
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from config import get_config
from research.research_engine import ResearchEngine
from research.topic_discovery import suggest_research_aspects
from utils.visualization import create_knowledge_graph, create_source_pie_chart

# Configure the application
config = get_config()
os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.reports_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

# Initialize the session state
if 'research_status' not in st.session_state:
    st.session_state.research_status = 'idle'
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'topics' not in st.session_state:
    st.session_state.topics = []
if 'selected_topics' not in st.session_state:
    st.session_state.selected_topics = []
if 'research_id' not in st.session_state:
    st.session_state.research_id = None

# App title and description
st.title("Personal Research Assistant")
st.markdown("""
This AI-powered research assistant helps you discover, collect, verify, and synthesize 
information on any topic of interest. Enter your topic below to get started.
""")

# Create the research engine instance
@st.cache_resource
def get_research_engine():
    return ResearchEngine()

research_engine = get_research_engine()

# Topic input section
with st.form("topic_form"):
    main_topic = st.text_input("Research Topic", placeholder="Enter your research topic...")
    depth_level = st.select_slider(
        "Research Depth",
        options=["Basic", "Standard", "Comprehensive", "Expert"],
        value="Standard",
        help="Determines how deep the research will go"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        include_academic = st.checkbox("Include Academic Sources", value=True)
    with col2:
        include_news = st.checkbox("Include News Articles", value=True)
    
    submitted = st.form_submit_button("Start Topic Discovery")
    
    if submitted and main_topic:
        with st.spinner("Discovering research aspects..."):
            topics = suggest_research_aspects(main_topic, depth_level)
            st.session_state.topics = topics
            st.session_state.research_status = 'topic_discovery'

# Topic selection section
if st.session_state.research_status == 'topic_discovery':
    st.subheader("Select Research Aspects")
    st.markdown(f"The following aspects of **{main_topic}** are recommended for a {depth_level.lower()} research:")
    
    topics_df = pd.DataFrame({
        "Aspect": [topic["title"] for topic in st.session_state.topics],
        "Description": [topic["description"] for topic in st.session_state.topics],
        "Include": [True for _ in st.session_state.topics]
    })
    
    edited_df = st.data_editor(
        topics_df,
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Select aspects to include in your research",
                default=True,
            )
        },
        disabled=["Aspect", "Description"],
        hide_index=True,
    )
    
    if st.button("Start Research"):
        selected_topics = []
        for i, row in edited_df.iterrows():
            if row["Include"]:
                selected_topics.append(st.session_state.topics[i])
        
        st.session_state.selected_topics = selected_topics
        st.session_state.research_id = f"{main_topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.research_status = 'in_progress'
        st.experimental_rerun()

# Research progress section
if st.session_state.research_status == 'in_progress':
    st.subheader("Research in Progress")
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Start research process
    try:
        with st.spinner("Researching..."):
            research_results = research_engine.run_research_flow(
                main_topic=main_topic,
                subtopics=st.session_state.selected_topics,
                depth_level=depth_level,
                include_academic=include_academic,
                include_news=include_news,
                research_id=st.session_state.research_id,
                progress_callback=lambda stage, progress: (
                    progress_placeholder.progress(progress),
                    status_placeholder.text(f"Stage: {stage}")
                )
            )
            st.session_state.research_results = research_results
            st.session_state.research_status = 'completed'
            st.experimental_rerun()
    except Exception as e:
        st.error(f"An error occurred during research: {str(e)}")
        st.session_state.research_status = 'idle'

# Research results section
if st.session_state.research_status == 'completed' and st.session_state.research_results:
    results = st.session_state.research_results
    st.subheader("Research Results")
    
    # Report tab, sources tab, visualizations tab
    tab1, tab2, tab3 = st.tabs(["Report", "Sources", "Visualizations"])
    
    with tab1:
        st.markdown(f"# {results['title']}")
        st.markdown(results['summary'])
        
        for section in results['sections']:
            st.markdown(f"## {section['title']}")
            st.markdown(section['content'])
        
        st.markdown("## References")
        for i, ref in enumerate(results['references']):
            st.markdown(f"{i+1}. {ref}")
        
        # Download report button
        report_path = f"{config.reports_dir}/{st.session_state.research_id}.md"
        with open(report_path, "w") as f:
            f.write(f"# {results['title']}\n\n")
            f.write(f"{results['summary']}\n\n")
            for section in results['sections']:
                f.write(f"## {section['title']}\n\n")
                f.write(f"{section['content']}\n\n")
            f.write("## References\n\n")
            for i, ref in enumerate(results['references']):
                f.write(f"{i+1}. {ref}\n")
        
        with open(report_path, "rb") as file:
            st.download_button(
                "Download Full Report",
                file,
                file_name=f"{st.session_state.research_id}.md",
                mime="text/markdown"
            )
    
    with tab2:
        sources_df = pd.DataFrame(results['sources'])
        st.dataframe(
            sources_df,
            column_config={
                "title": "Title",
                "url": st.column_config.LinkColumn("URL"),
                "type": "Source Type",
                "reliability_score": st.column_config.ProgressColumn(
                    "Reliability Score",
                    min_value=0,
                    max_value=1
                )
            },
            hide_index=True
        )
    
    with tab3:
        st.subheader("Knowledge Graph")
        knowledge_graph = create_knowledge_graph(results)
        st.pyplot(knowledge_graph)
        
        st.subheader("Source Distribution")
        fig = create_source_pie_chart(results['sources'])
        st.plotly_chart(fig)
    
    # Start a new research button
    if st.button("Start New Research"):
        st.session_state.research_status = 'idle'
        st.session_state.research_results = None
        st.session_state.topics = []
        st.session_state.selected_topics = []
        st.session_state.research_id = None
        st.experimental_rerun()

if __name__ == "__main__":
    # This is used when running the file directly
    pass
