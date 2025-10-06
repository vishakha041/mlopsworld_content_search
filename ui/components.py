"""
Reusable UI Components for Streamlit Interface

This module contains all reusable UI components and the agent
integration logic with streaming support.
"""

import streamlit as st
from typing import Dict, Any, List
import json

from .state import (
    get_state,
    set_state,
    clear_chat_history,
    add_message,
    add_agent_step,
    clear_agent_steps
)
from .examples import EXAMPLE_QUERIES


def render_header():
    """Render the app header and description."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 MLOps Events Agent</h1>
            <p style="color: #666; font-size: 1.1rem;">
                Powered by LangGraph + ApertureDB + Gemini 2.5 Pro
            </p>
            <p style="color: #888; font-size: 0.95rem;">
                Query 278 MLOps & GenAI conference talks with natural language
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_example_queries():
    """Render example query buttons organized by category."""
    st.markdown("### 💡 Example Queries")
    st.markdown("Click any example to try it out:")
    
    # Create columns for categories
    cols = st.columns(len(EXAMPLE_QUERIES))
    
    for idx, (category_key, category_data) in enumerate(EXAMPLE_QUERIES.items()):
        with cols[idx]:
            st.markdown(f"**{category_data['label']}**")
            for q_idx, query in enumerate(category_data['queries']):
                if st.button(
                    query,
                    key=f"example_{category_key}_{q_idx}",
                    use_container_width=True
                ):
                    # Set the query in session state (don't increment counter to preserve input)
                    st.session_state.pending_example_query = query
                    st.rerun()


def render_chat_interface():
    """
    Render the chat interface with input and message display.
    
    Returns:
        str or None: The submitted query, if any
    """
    # Initialize the query value in session state if not exists
    query_key = f"query_input_{st.session_state.input_key_counter}"
    if query_key not in st.session_state:
        st.session_state[query_key] = ""
    
    # Handle example query click - directly set the session state value
    if "pending_example_query" in st.session_state and st.session_state.pending_example_query:
        st.session_state[query_key] = st.session_state.pending_example_query
        st.session_state.pending_example_query = None
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_area(
            "Ask a question:",
            placeholder="e.g., Show me talks about deployment strategies...",
            height=100,
            key=query_key,
            label_visibility="collapsed"
        )
    
    with col2:
        submit = st.button("🚀 Submit", type="primary", use_container_width=True)
        if st.button("🗑️ Clear", use_container_width=True):
            clear_chat_history()
            st.session_state.current_query = ""
            st.session_state.input_key_counter += 1
            st.rerun()
    
    # Display chat messages
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    
    if not st.session_state.messages:
        st.info("👋 Ask a question to get started! Try one of the examples above.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Handle submit
    if submit and query.strip() and not st.session_state.is_processing:
        # Store query in session state for processing
        st.session_state.pending_query = query.strip()
        # Clear chat history immediately
        clear_chat_history()
        st.session_state.current_query = ""
        # Increment counter to reset input field
        st.session_state.input_key_counter += 1
        st.rerun()
    
    return None


def render_agent_steps_panel():
    """Render the agent execution steps panel."""
    if not st.session_state.agent_steps:
        return
    
    with st.expander("🔍 **Agent Execution Steps**", expanded=st.session_state.show_steps):
        st.markdown("*Agent's reasoning and tool usage:*")
        st.markdown("")
        
        for idx, step in enumerate(st.session_state.agent_steps, 1):
            step_type = step.get("type", "unknown")
            
            if step_type == "tool_call":
                tool_name = step.get("tool_name", "Unknown")
                tool_args = step.get("tool_args", {})
                
                st.markdown(f"**Step {idx}: Tool Call**")
                st.code(f"🔧 Tool: {tool_name}", language=None)
                
                if tool_args:
                    st.markdown("*Arguments:*")
                    st.json(tool_args)
            
            elif step_type == "tool_result":
                result = step.get("result", "")
                result_str = str(result)
                
                st.markdown(f"**Step {idx}: Tool Result**")
                if len(result_str) > 5000:
                    st.text(result_str[:5000] + "...")
                else:
                    st.text(result_str)
            
            st.markdown("---")


def run_agent_with_streaming(query: str) -> str:
    """
    Run the agent with the given query and stream updates to UI.
    
    This function wraps the existing agent.query_agent function and
    adapts its streaming output for Streamlit. It captures agent
    execution steps and displays them in real-time.
    
    Args:
        query: User's natural language query
        
    Returns:
        str: The final agent response
    """
    from agent.agent import create_mlops_agent
    import time
    
    # Clear previous steps
    clear_agent_steps()
    
    # Create placeholders for streaming response and dynamic status
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    current_response = ""
    
    # Helper function to update status message
    def update_status(message: str):
        st.session_state.agent_status = message
        with status_placeholder.container():
            st.info(f"⏳ {message}")
    
    # Helper function to render execution steps (only shown after completion)
    def show_execution_steps():
        if st.session_state.agent_steps:
            with st.expander("🔍 **Agent Execution Steps**", expanded=False):
                st.markdown("*Agent's reasoning and tool usage:*")
                st.markdown("")
                
                for idx, step in enumerate(st.session_state.agent_steps, 1):
                    step_type = step.get("type", "unknown")
                    
                    if step_type == "tool_call":
                        tool_name = step.get("tool_name", "Unknown")
                        tool_args = step.get("tool_args", {})
                        
                        st.markdown(f"**Step {idx}: Tool Call**")
                        st.code(f"🔧 Tool: {tool_name}", language=None)
                        
                        if tool_args:
                            st.markdown("*Arguments:*")
                            st.json(tool_args)
                    
                    elif step_type == "tool_result":
                        result = step.get("result", "")
                        result_str = str(result)
                        
                        st.markdown(f"**Step {idx}: Tool Result**")
                        if len(result_str) > 5000:
                            st.text(result_str[:5000] + "...")
                        else:
                            st.text(result_str)
                    
                    st.markdown("---")
    
    # Create agent
    try:
        update_status("Initializing agent...")
        agent = create_mlops_agent()
    except Exception as e:
        return f"❌ Error creating agent: {str(e)}"
    
    # Prepare input
    inputs = {"messages": [{"role": "user", "content": query}]}
    
    # Stream the agent's execution
    step_count = 0
    final_response = None
    current_tool_name = None
    
    try:
        update_status("Thinking which tool to call...")
        
        for event in agent.stream(inputs, stream_mode="values"):
            step_count += 1
            
            # Get the last message in the current state
            messages = event.get("messages", [])
            if not messages:
                continue
            
            last_message = messages[-1]
            
            # Handle different message types
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Tool call step
                for tool_call in last_message.tool_calls:
                    current_tool_name = tool_call['name']
                    add_agent_step({
                        "type": "tool_call",
                        "status": "complete",
                        "tool_name": current_tool_name,
                        "tool_args": tool_call['args'],
                        "call_id": tool_call['id']
                    })
                    
                    # Update status with tool name
                    update_status(f"Tool decided - {current_tool_name}")
                    time.sleep(1.0)  # Pause to make status visible
                    update_status("ApertureDB query running...")
                    time.sleep(1.0)
            
            # Check for content (response or tool result)
            if hasattr(last_message, 'content') and last_message.content:
                content = last_message.content
                
                # Check if this is a tool result message
                if "ToolMessage" in str(type(last_message)):
                    result_content = content[:5000]
                    add_agent_step({
                        "type": "tool_result",
                        "status": "complete",
                        "result": result_content
                    })
                    
                    # Store the LAST tool result for sidebar display
                    # Import here to avoid circular dependency
                    from .sidebar import update_sidebar_results
                    update_sidebar_results(content)
                    
                    # Update status
                    update_status("Got results from ApertureDB - thinking...")
                    
                else:
                    # This is the final AI response - stream it
                    current_response = content
                    status_placeholder.empty()  # Clear status
                    with response_placeholder.container():
                        with st.chat_message("assistant"):
                            st.markdown(current_response)
            
            # Store final response
            final_response = event
        
        # Clear status and show execution steps dropdown (collapsed)
        status_placeholder.empty()
        with status_placeholder.container():
            show_execution_steps()
        
        # Extract final answer
        if final_response and "messages" in final_response:
            last_msg = final_response["messages"][-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
        
        return "No response generated."
    
    except Exception as e:
        error_msg = f"❌ Error during agent execution: {str(e)}"
        add_agent_step({
            "type": "error",
            "status": "error",
            "content": str(e)
        })
        status_placeholder.empty()
        return error_msg


def show_info_message(message: str, icon: str = "ℹ️"):
    """Display an info message."""
    st.info(f"{icon} {message}")


def show_error_message(message: str):
    """Display an error message."""
    st.error(f"❌ {message}")


def show_success_message(message: str):
    """Display a success message."""
    st.success(f"✅ {message}")
