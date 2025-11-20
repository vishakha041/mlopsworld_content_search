"""
Video Semantic Search UI Component

This module provides the UI components for direct video semantic search
using Twelve Labs embeddings. Separate from the chat-based agent interface.
"""

import streamlit as st
import tempfile
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)


# ===== EXAMPLE QUERIES =====
VIDEO_SEARCH_EXAMPLES = {
    "search_1": {
        "label": "🔍 Content Search",
        "queries": [
            "AI agents and LLMs in production",
            "Vector databases and RAG systems"
        ]
    },
    "search_2": {
        "label": "🎯 Topics",
        "queries": [
            "MLOps best practices and deployment",
            "Model monitoring and evaluation"
        ]
    },
    "search_3": {
        "label": "🤖 Technologies",
        "queries": [
            "Data quality and feature engineering",
            "Generative AI and prompt engineering"
        ]
    }
}


# ===== HELPER FUNCTIONS =====

def get_youtube_thumbnail(youtube_id: str) -> str:
    """Get YouTube thumbnail URL from video ID."""
    return f"https://img.youtube.com/vi/{youtube_id}/mqdefault.jpg"


def display_video_blob(video_blob):
    """
    Display video blob using Streamlit's video player.
    
    Args:
        video_blob (bytes): Video data as bytes
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Try direct display first
        st.video(video_blob)
        return True
    except Exception as e:
        logging.warning(f"Direct video display failed: {e}")
        
        try:
            # Fallback to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_blob)
                tmp_file_path = tmp_file.name
            
            st.video(tmp_file_path)
            return True
            
        except Exception as e2:
            logging.error(f"Temp file video display failed: {e2}")
            
            # Last resort: download button
            st.download_button(
                label="⬇️ Download Video",
                data=video_blob,
                file_name="video.mp4",
                mime="video/mp4"
            )
            return False


def render_video_result_card(result: Dict[str, Any], index: int):
    """
    Render a single video search result as a card.
    
    Args:
        result: Result dictionary with video metadata
        index: Result index for display
    """
    st.markdown(f"### 🎯 Match #{index}: {result.get('talk_title', 'Untitled')}")
    st.markdown("---")
    # Create columns for metadata
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        speaker = result.get('speaker_name', 'Unknown')
        company = result.get('company_name', '')
        if company:
            st.markdown(f"**Speaker:** {speaker} ({company})")
        else:
            st.markdown(f"**Speaker:** {speaker}")
    
    with col2:
        similarity = result.get('similarity_score', 0)
        st.markdown(f"**Similarity:** {similarity:.1%}")
    
    with col3:
        distance = result.get('distance', 0)
        st.caption(f"Distance: {distance:.4f}")
    
    # Display additional metadata
    metadata = result.get('metadata', {})
    if metadata:
        cols = st.columns(4)
        
        with cols[0]:
            fps = metadata.get("fps", 0)
            if fps:
                st.metric("FPS", f"{fps:.2f}")
        
        with cols[1]:
            duration = metadata.get("duration_seconds", 0)
            if duration:
                st.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s")
        
        with cols[2]:
            height = metadata.get("frame_height", 0)
            if height:
                st.metric("Height", f"{height}p")
        
        with cols[3]:
            width = metadata.get("frame_width", 0)
            if width:
                st.metric("Width", f"{width}px")
    
    # YouTube link if available
    youtube_url = result.get('youtube_url', '')
    youtube_id = result.get('youtube_id', '')
    
    if youtube_id:
        thumbnail_url = get_youtube_thumbnail(youtube_id)
        if youtube_url:
            st.markdown(
                f'<a href="{youtube_url}" target="_blank">'
                f'<img src="{thumbnail_url}" width="100%" style="border-radius: 8px; max-width: 480px;"/>'
                f'</a>',
                unsafe_allow_html=True
            )
    
    # Display video blob if available
    if 'video_blob' in result and result['video_blob']:
        st.markdown("#### 🎬 Video Player")
        st.caption("Tip: Use fullscreen mode for better viewing")
        display_video_blob(result['video_blob'])
    
    # Add separator
    st.markdown("---")


# ===== MAIN COMPONENT =====

def render_video_search_tab():
    """
    Render the video semantic search tab content.
    
    This provides a direct interface for searching videos using
    Twelve Labs embeddings, separate from the chat-based agent.
    """
    
    # Header
    st.markdown("### 🎥 Video Semantic Search")
    st.caption("Uses Twelve Labs Marengo video embeddings to search video content")
    
    # Initialize session state for video search
    if "video_search_query" not in st.session_state:
        st.session_state.video_search_query = ""
    if "video_search_results" not in st.session_state:
        st.session_state.video_search_results = None
    if "video_input_key_counter" not in st.session_state:
        st.session_state.video_input_key_counter = 0
    if "pending_video_example_query" not in st.session_state:
        st.session_state.pending_video_example_query = None
    
    # Create two columns: search controls on left, examples on right
    col1, col2 = st.columns([2, 2])
    
    with col1:
        # Dynamic key for input field (same pattern as chat interface)
        video_query_key = f"video_query_input_{st.session_state.video_input_key_counter}"
        if video_query_key not in st.session_state:
            st.session_state[video_query_key] = ""
        
        # Handle example query click - directly set the session state value
        if st.session_state.pending_video_example_query:
            st.session_state[video_query_key] = st.session_state.pending_video_example_query
            st.session_state.pending_video_example_query = None
        
        search_query = st.text_input(
            "Search Query:",
            placeholder="e.g., 'AI agents with memory and context'",
            key=video_query_key,
            label_visibility="visible"
        )
        
        # Number of results slider and checkbox
        slider_col, check_col = st.columns([3, 1])
        
        with slider_col:
            top_n = st.slider(
                "Number of results:",
                min_value=1,
                max_value=10,
                value=3,
                help="Select how many matching videos to display"
            )
        
        with check_col:
            include_videos = st.checkbox(
                "Show videos",
                value=True,
                help="Include video players in results (disable for faster metadata-only search)"
            )
        
        # Search button
        search_clicked = st.button("🔍 Search Videos", type="primary", use_container_width=True)
    
    with col2:
        # Example queries in dropdown
        with st.expander("💡 Example Queries", expanded=False):
            # Create columns for categories
            cols = st.columns(len(VIDEO_SEARCH_EXAMPLES))
            
            for idx, (category_key, category_data) in enumerate(VIDEO_SEARCH_EXAMPLES.items()):
                with cols[idx]:
                    st.markdown(f"**{category_data['label']}**")
                    for q_idx, query in enumerate(category_data['queries']):
                        if st.button(
                            query,
                            key=f"video_example_{category_key}_{q_idx}",
                            use_container_width=True
                        ):
                            st.session_state.pending_video_example_query = query
                            st.rerun()
    
    # Perform search when button is clicked
    if search_clicked and search_query.strip():
        # Update session state
        st.session_state.video_search_query = search_query
        
        # Import the tool here to avoid circular imports
        from tools.search_videos_semantically import search_videos_semantically
        
        # Show searching status
        with st.spinner(f"🔍 Searching for '{search_query}'..."):
            # Call the tool directly
            response = search_videos_semantically.invoke({
                "query": search_query,
                "top_n": top_n,
                "return_blobs": include_videos
            })
        
        # Store results in session state
        st.session_state.video_search_results = response
    
    elif search_clicked and not search_query.strip():
        st.warning("⚠️ Please enter a search query")
    
    # Display results if available
    if st.session_state.video_search_results:
        response = st.session_state.video_search_results
        
        if response.get("success") and response.get("results"):
            results = response["results"]
            total = response.get("total_found", len(results))
            
            st.success(f"✅ Found {total} matching video{'s' if total != 1 else ''}")
            st.caption(response.get("search_summary", ""))
            st.markdown("---")
            
            # Display each result
            for i, result in enumerate(results, 1):
                render_video_result_card(result, i)
        
        elif response.get("success") and not response.get("results"):
            st.warning("No matching videos found. Try a different query or broader search terms.")
        
        else:
            error_msg = response.get("error", "Unknown error occurred")
            st.error(f"❌ Search failed: {error_msg}")
    
    # Footer with helpful tips
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        <p><strong>💡 Search Tips:</strong></p>
        <p>• Use natural language queries for best results</p>
        <p>• The search understands context and semantic meaning of video content</p>
        <p>• Lower distance values indicate better matches</p>
        <p>• Similarity scores range from 0% (not similar) to 100% (highly similar)</p>
    </div>
    """, unsafe_allow_html=True)
