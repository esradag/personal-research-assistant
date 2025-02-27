"""
LangChain helper functions for the Personal Research Assistant.
"""
import logging
from typing import Optional

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from config import get_config
from utils.llm_utils import get_llm

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def get_memory() -> ConversationBufferMemory:
    """
    Get a LangChain conversation memory instance.
    
    Returns:
        ConversationBufferMemory: LangChain conversation memory
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def get_conversation_chain(memory: Optional[ConversationBufferMemory] = None) -> ConversationChain:
    """
    Get a LangChain conversation chain.
    
    Args:
        memory: Memory to use in the chain
        
    Returns:
        ConversationChain: LangChain conversation chain
    """
    llm = get_llm()
    memory = memory or get_memory()
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def create_research_tool_chain(tool_name: str, tool_description: str, llm_wrapper):
    """
    Create a research tool chain with LangChain.
    
    Args:
        tool_name: Name of the tool
        tool_description: Description of the tool
        llm_wrapper: Function wrapper for tool execution
        
    Returns:
        Tool: LangChain tool
    """
    from langchain.tools import Tool
    
    return Tool(
        name=tool_name,
        description=tool_description,
        func=llm_wrapper
    )

def create_research_agent(tools, prefix="Research", suffix="Begin!", verbose=True):
    """
    Create a research agent with LangChain.
    
    Args:
        tools: List of tools for the agent
        prefix: Prefix for the agent prompt
        suffix: Suffix for the agent prompt
        verbose: Whether to show verbose output
        
    Returns:
        AgentExecutor: LangChain agent executor
    """
    from langchain.agents import initialize_agent, AgentType
    
    llm = get_llm()
    
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix
        }
    )
