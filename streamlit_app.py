"""
Streamlit Web Interface for MLOps Events Agent

This is the main entry point for the Streamlit application.
It provides a web-based UI for querying the MLOps Events database
using the LangGraph agent.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st

# Import UI components
from ui import (
    initialize_session_state,
    render_header,
    render_example_queries,
    render_chat_interface,
    render_agent_steps_panel,
    run_agent_with_streaming,
    get_custom_css,
    render_results_sidebar
)
from ui.state import add_message, set_state


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="MLOps Events Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"  # Changed from "collapsed" to allow auto-open
)


# ===== INITIALIZE SESSION STATE =====
initialize_session_state()


# ===== APPLY CUSTOM CSS =====
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ===== MAIN APP LAYOUT =====
def main():
    """Main application function."""
    
    # Render header
    render_header()
    
    # Render sidebar with results (if any)
    render_results_sidebar()
    
    # Render example queries
    with st.expander("💡 **Example Queries**", expanded=False):
        render_example_queries()
    
    # Render chat interface and get submitted query
    submitted_query = render_chat_interface()
    
    # Process query if submitted
    if submitted_query:
        # Set processing flag
        set_state("is_processing", True)
        
        # Add user message to chat
        add_message("user", submitted_query)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(submitted_query)
        
        # Run agent with streaming
        with st.spinner("🤖 Agent is thinking..."):
            response = run_agent_with_streaming(submitted_query)
        
        # Add assistant response to chat
        add_message("assistant", response)
        
        # Reset processing flag
        set_state("is_processing", False)
        
        # Rerun to update UI
        st.rerun()
    
    # Render agent steps panel (if any steps exist)
    render_agent_steps_panel()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85rem;">
            <p>Built with ❤️ using Streamlit, LangGraph, and ApertureDB</p>
            <p>Dataset: 278 MLOps & GenAI conference talks</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ===== RUN APP =====
if __name__ == "__main__":
    main()
