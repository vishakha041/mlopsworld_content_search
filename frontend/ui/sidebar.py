"""
Sidebar Results Display for Streamlit UI

This module handles the display of tool results in the sidebar
as interactive cards with YouTube thumbnails and metadata.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional


def extract_results_from_tool_output(tool_result: str) -> Optional[List[Dict]]:
    """
    Extract structured results from tool output.
    
    Args:
        tool_result: Raw tool result string (usually JSON)
        
    Returns:
        List of result dictionaries, or None if extraction fails
    """
    try:
        # Try to parse as JSON
        parsed = json.loads(tool_result)
        
        # Check if it's a dictionary with results
        if isinstance(parsed, dict):
            # Pattern 1: {"success": true, "results": [...]}
            if "results" in parsed:
                results = parsed["results"]
                if isinstance(results, list) and len(results) > 0:
                    print(f"✅ Extracted {len(results)} results from tool output")
                    return results
            
            # Pattern 2: {"success": true, "talks": [...]}
            if "talks" in parsed:
                talks = parsed["talks"]
                if isinstance(talks, list) and len(talks) > 0:
                    print(f"✅ Extracted {len(talks)} talks from tool output")
                    return talks
            
            # Pattern 3: {"speakers": [...], ...}
            if "speakers" in parsed:
                speakers = parsed["speakers"]
                if isinstance(speakers, list) and len(speakers) > 0:
                    print(f"✅ Extracted {len(speakers)} speakers from tool output")
                    return speakers
            
            # Pattern 4: Single talk object {"talk_title": "...", "youtube_url": "...", ...}
            if "talk_title" in parsed or "youtube_url" in parsed or "title" in parsed:
                print(f"✅ Extracted single talk from tool output")
                return [parsed]
        
        # Pattern 5: If it's already a list
        elif isinstance(parsed, list) and len(parsed) > 0:
            print(f"✅ Extracted {len(parsed)} items from tool output (list format)")
            return parsed
        
        print(f"⚠️ No results found in tool output. Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
        return None
    except (json.JSONDecodeError, TypeError) as e:
        # Log the error for debugging
        print(f"❌ Failed to parse tool result: {e}")
        print(f"Tool result preview: {tool_result[:200] if len(tool_result) > 200 else tool_result}")
        return None


def get_youtube_thumbnail(youtube_id: str) -> str:
    """
    Get YouTube thumbnail URL from video ID.
    
    Args:
        youtube_id: YouTube video ID
        
    Returns:
        Thumbnail URL
    """
    return f"https://img.youtube.com/vi/{youtube_id}/mqdefault.jpg"


def render_result_card(result: Dict[str, Any], index: int):
    """
    Render a single result as a card in the sidebar.
    
    Args:
        result: Result dictionary with talk metadata
        index: Card index for display
    """
    # Extract fields with safe defaults - supporting multiple field name formats
    title = result.get("title") or result.get("talk_title", "Untitled Talk")
    youtube_url = result.get("youtube_url", "")
    youtube_id = result.get("youtube_id", "")
    speaker = result.get("speaker") or result.get("speaker_name", "Unknown Speaker")
    company = result.get("company") or result.get("company_name", "")
    views = result.get("views") or result.get("yt_views", 0)
    event = result.get("event") or result.get("event_name", "")
    category = result.get("category") or result.get("category_primary", "")
    
    # Extract YouTube ID from URL if not provided directly
    if not youtube_id and youtube_url:
        # Extract from various YouTube URL formats
        if "youtube.com/watch?v=" in youtube_url:
            youtube_id = youtube_url.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            youtube_id = youtube_url.split("youtu.be/")[-1].split("?")[0]
    
    # Card container in sidebar
    with st.sidebar.container():
        # YouTube thumbnail (if available)
        if youtube_id:
            thumbnail_url = get_youtube_thumbnail(youtube_id)
            if youtube_url:
                st.markdown(
                    f'<a href="{youtube_url}" target="_blank">'
                    f'<img src="{thumbnail_url}" width="100%" style="border-radius: 8px;"/>'
                    f'</a>',
                    unsafe_allow_html=True
                )
        
        # Title (linked to YouTube if available)
        if youtube_url:
            st.markdown(f"**[{title}]({youtube_url})**")
        else:
            st.markdown(f"**{title}**")
        
        # Metadata
        metadata_parts = []
        
        if speaker and speaker != "Unknown Speaker":
            metadata_parts.append(f"👤 {speaker}")
        
        if company:
            metadata_parts.append(f"🏢 {company}")
        
        if views and views > 0:
            metadata_parts.append(f"👁️ {views:,} views")
        
        if metadata_parts:
            st.caption(" • ".join(metadata_parts))
        
        # Additional info
        info_parts = []
        if event:
            info_parts.append(f"📅 {event}")
        if category:
            info_parts.append(f"🏷️ {category}")
        
        if info_parts:
            st.caption(" • ".join(info_parts))
        
        # Separator
        st.markdown("---")


def render_results_sidebar():
    """
    Render the results sidebar with cards.
    
    This function is called from the main app to display
    the last tool call's results in the sidebar.
    """
    # Safely check if last_tool_results exists and has content
    if "last_tool_results" not in st.session_state or not st.session_state.last_tool_results:
        # Sidebar is empty or no results yet
        st.sidebar.info("👋 Results will appear here after you run a query")
        return
    
    # Render header
    st.sidebar.markdown("### 📊 Retrieved Results")
    st.sidebar.caption("Here are some of the results from your query")
    st.sidebar.markdown("")
    
    # Get results (limit to first 10)
    results = st.session_state.last_tool_results[:10]
    
    if not results:
        st.sidebar.info("No results to display")
        return
    
    # Display count
    total = len(st.session_state.last_tool_results)
    if total > 10:
        st.sidebar.caption(f"Showing 10 of {total} results")
    else:
        st.sidebar.caption(f"Showing {total} result{'s' if total != 1 else ''}")
    
    st.sidebar.markdown("---")
    
    # Render each result as a card
    for idx, result in enumerate(results, 1):
        render_result_card(result, idx)
    
    # Footer
    if total > 10:
        st.sidebar.info(f"💡 {total - 10} more results available. Try refining your query!")


def update_sidebar_results(tool_result: str):
    """
    Update the sidebar with new tool results.
    
    Args:
        tool_result: Raw tool result string from the last tool call
    """
    # Extract structured results
    results = extract_results_from_tool_output(tool_result)
    
    if results:
        st.session_state.last_tool_results = results
    else:
        # If we can't extract results, clear the sidebar
        st.session_state.last_tool_results = None
