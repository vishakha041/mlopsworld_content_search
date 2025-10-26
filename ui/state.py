"""
Session State Management for Streamlit UI

This module handles all Streamlit session state initialization
and access patterns.
"""

import streamlit as st


def initialize_session_state():
    """
    Initialize all session state variables.
    
    This should be called once at the start of the Streamlit app.
    """
    # Chat messages history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Agent execution steps for current query
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    
    # Current query text
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Pending query waiting to be processed
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    
    # Input field key counter (increment to reset input)
    if "input_key_counter" not in st.session_state:
        st.session_state.input_key_counter = 0
    
    # Processing flag to prevent double submissions
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # Toggle for showing agent steps panel
    if "show_steps" not in st.session_state:
        st.session_state.show_steps = False
    
    # Last tool results for sidebar display
    if "last_tool_results" not in st.session_state:
        st.session_state.last_tool_results = None
    
    # Dynamic status message for agent execution
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = ""
    
    # ApertureDB connection pool (for session-level persistence)
    if "db_connector" not in st.session_state:
        st.session_state.db_connector = None
    
    # Embedding model (for session-level persistence)
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    
    # Twelve Labs client (for session-level persistence)
    if "twelvelabs_client" not in st.session_state:
        st.session_state.twelvelabs_client = None
    
    # Video search specific state
    if "video_search_query" not in st.session_state:
        st.session_state.video_search_query = ""
    
    if "video_search_results" not in st.session_state:
        st.session_state.video_search_results = None
    
    if "video_input_key_counter" not in st.session_state:
        st.session_state.video_input_key_counter = 0
    
    if "pending_video_example_query" not in st.session_state:
        st.session_state.pending_video_example_query = None
    
    # Expander states for auto-closing
    if "chat_examples_expanded" not in st.session_state:
        st.session_state.chat_examples_expanded = False
    
    if "video_examples_expanded" not in st.session_state:
        st.session_state.video_examples_expanded = False


def get_state(key: str, default=None):
    """
    Safely get a session state value.
    
    Args:
        key: Session state key
        default: Default value if key doesn't exist
        
    Returns:
        The session state value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value):
    """
    Set a session state value.
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def clear_chat_history():
    """Clear all chat messages and agent steps."""
    st.session_state.messages = []
    st.session_state.agent_steps = []
    st.session_state.current_query = ""


def add_message(role: str, content: str):
    """
    Add a message to chat history.
    
    Args:
        role: Either "user" or "assistant"
        content: Message content
    """
    st.session_state.messages.append({
        "role": role,
        "content": content
    })


def add_agent_step(step_data: dict):
    """
    Add an agent execution step.
    
    Args:
        step_data: Dictionary containing step information
    """
    st.session_state.agent_steps.append(step_data)


def clear_agent_steps():
    """Clear agent steps for new query."""
    st.session_state.agent_steps = []
