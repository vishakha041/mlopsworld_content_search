"""
Streamlit UI Module for MLOps Events Agent

This module contains all UI components, state management,
and styling for the Streamlit web interface.
"""

from .state import initialize_session_state, get_state, set_state
from .examples import EXAMPLE_QUERIES, get_all_examples
from .components import (
    render_header,
    render_example_queries,
    render_chat_interface,
    render_agent_steps_panel,
    run_agent_with_streaming
)
from .styles import get_custom_css
from .sidebar import render_results_sidebar
from .video_search import render_video_search_tab

__all__ = [
    "initialize_session_state",
    "get_state",
    "set_state",
    "EXAMPLE_QUERIES",
    "get_all_examples",
    "render_header",
    "render_example_queries",
    "render_chat_interface",
    "render_agent_steps_panel",
    "run_agent_with_streaming",
    "get_custom_css",
    "render_results_sidebar",
    "render_video_search_tab"
]
