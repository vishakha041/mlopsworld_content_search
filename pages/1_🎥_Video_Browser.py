"""
Video Browser Page

A simple interface to browse and watch MLOps conference talk videos
stored in ApertureDB. Select a talk from the dropdown to view its video.
"""

import streamlit as st
import sys
import os
import tempfile
import base64

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils import get_db_connector


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Video Browser - MLOps Events",
    page_icon="🎥",
    layout="wide"
)


# ===== DATA FETCHING FUNCTIONS =====

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_all_talks():
    """
    Fetch all unique talks with their titles and speaker names from ApertureDB.
    
    Returns:
        list: List of dictionaries with 'talk_title' and 'speaker_name' keys
    """
    try:
        client = get_db_connector()
        
        # Query to get all Video entities with talk_title and speaker_name
        query = [{
            "FindVideo": {
                "results": {
                    "list": ["talk_title", "speaker_name"]
                }
            }
        }]
        
        response, _ = client.query(query)
        
        # Check if query was successful
        if response[0]["FindVideo"]["status"] != 0:
            st.error("Failed to fetch talks from database")
            return []
        
        # Extract entities
        entities = response[0]["FindVideo"].get("entities", [])
        
        # Create list of talks with titles and speakers
        talks = []
        seen_titles = set()  # To track unique talks
        
        for entity in entities:
            talk_title = entity.get("talk_title")
            speaker_name = entity.get("speaker_name")
            
            # Only add if we have a talk title and haven't seen it before
            if talk_title and talk_title not in seen_titles:
                talks.append({
                    "talk_title": talk_title,
                    "speaker_name": speaker_name or "Unknown Speaker"
                })
                seen_titles.add(talk_title)
        
        # Sort by talk title for easier browsing
        talks.sort(key=lambda x: x["talk_title"])
        
        return talks
        
    except Exception as e:
        st.error(f"Error fetching talks: {str(e)}")
        return []


def fetch_video_by_talk(talk_title):
    """
    Fetch video blob and metadata for a specific talk.
    
    Args:
        talk_title (str): The title of the talk to fetch
        
    Returns:
        tuple: (video_blob, metadata_dict) or (None, None) if not found
    """
    try:
        client = get_db_connector()
        
        # Query to find video by talk title and get the blob
        query = [{
            "FindVideo": {
                "constraints": {
                    "talk_title": ["==", talk_title]
                },
                "results": {
                    "list": [
                        "talk_title", 
                        "speaker_name",
                        "_fps",
                        "_frame_count",
                        "_frame_height",
                        "_frame_width"
                    ]
                },
                "blobs": True
            }
        }]
        
        response, blobs = client.query(query)
        
        # Check if query was successful
        if response[0]["FindVideo"]["status"] != 0:
            st.error("Failed to fetch video from database")
            return None, None
        
        # Check if we found any videos
        num_videos = response[0]["FindVideo"].get("returned", 0)
        if num_videos == 0:
            st.warning(f"No video found for talk: {talk_title}")
            return None, None
        
        # Get the first video (should only be one per talk title)
        video_blob = blobs[0] if blobs else None
        metadata = response[0]["FindVideo"]["entities"][0] if response[0]["FindVideo"]["entities"] else {}
        
        return video_blob, metadata
        
    except Exception as e:
        st.error(f"Error fetching video: {str(e)}")
        return None, None


# ===== VIDEO DISPLAY FUNCTIONS =====

def display_video(video_blob):
    """
    Display video blob using Streamlit's video player.
    
    This function handles the video blob by creating a temporary file
    and using Streamlit's native video player.
    
    Args:
        video_blob (bytes): The video blob data from ApertureDB
        
    Returns:
        bool: True if video displayed successfully, False otherwise
    """
    try:
        # Method 1: Try displaying video directly from bytes (simplest approach)
        try:
            st.video(video_blob)
            return True
        except Exception as e:
            st.warning("Direct video display failed, trying alternative method...")
        
        # Method 2: Save to temporary file and display
        # Create a temporary file with .mp4 extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_blob)
            tmp_file_path = tmp_file.name
        
        # Display the video using the temporary file path
        st.video(tmp_file_path)
        
        # Clean up the temporary file after a delay
        # Note: We can't delete immediately as Streamlit needs to read it
        # The OS will clean up temp files eventually
        
        return True
        
    except Exception as e:
        st.error(f"Error displaying video: {str(e)}")
        
        # Fallback: Show download option
        st.warning("Unable to display video in browser. You can download it instead:")
        
        # Create download button
        st.download_button(
            label="⬇️ Download Video",
            data=video_blob,
            file_name="conference_talk.mp4",
            mime="video/mp4"
        )
        
        return False


# ===== MAIN PAGE UI =====

def main():
    """Main function to render the Video Browser page."""
    
    # Header
    st.title("🎥 MLOps Conference Talk Videos")
    st.markdown("""
    Browse and watch videos from MLOps and GenAI conference talks stored in ApertureDB.
    Select a talk from the dropdown below to view the video.
    """)
    
    st.markdown("---")
    
    # Fetch all talks
    with st.spinner("Loading talks from database..."):
        talks = fetch_all_talks()
    
    # Check if we have any talks
    if not talks:
        st.error("No talks found in the database. Please check your connection.")
        return
    
    # Display talk count
    st.success(f"📊 Found **{len(talks)}** unique talks in the database")
    
    # Create dropdown options
    talk_options = [f"{talk['talk_title']} - {talk['speaker_name']}" for talk in talks]
    
    # Dropdown to select talk
    st.subheader("Select a Talk")
    selected_option = st.selectbox(
        "Choose a talk to watch:",
        options=["-- Select a talk --"] + talk_options,
        index=0
    )
    
    # If a talk is selected
    if selected_option != "-- Select a talk --":
        # Extract the talk title from the selection
        selected_talk_title = selected_option.split(" - ")[0]
        
        # Find the selected talk data
        selected_talk = next(
            (talk for talk in talks if talk["talk_title"] == selected_talk_title),
            None
        )
        
        if selected_talk:
            # Display speaker info
            st.markdown(f"**Speaker:** {selected_talk['speaker_name']}")
            
            # Fetch and display video
            st.markdown("---")
            st.subheader("Video")
            
            with st.spinner("Loading video..."):
                video_blob, metadata = fetch_video_by_talk(selected_talk_title)
            
            if video_blob and metadata:
                # Display video metadata
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    fps = metadata.get("_fps", 0)
                    st.metric("FPS", f"{fps:.2f}" if fps else "N/A")
                
                with col2:
                    frame_count = metadata.get("_frame_count", 0)
                    duration = frame_count / fps if fps else 0
                    st.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "N/A")
                
                with col3:
                    height = metadata.get("_frame_height", 0)
                    st.metric("Height", f"{height}p" if height else "N/A")
                
                with col4:
                    width = metadata.get("_frame_width", 0)
                    st.metric("Width", f"{width}px" if width else "N/A")
                
                st.markdown("---")
                
                # Display video
                st.markdown("### 🎬 Video Player")
                
                # Add a note about video quality
                st.caption("Note: Videos are displayed at their original resolution. Use fullscreen mode for better viewing.")
                
                # Display the video
                success = display_video(video_blob)
                
                if success:
                    # Add helpful information
                    st.markdown("---")
                    st.info("💡 **Tip**: Click the fullscreen icon (⛶) in the video player for better viewing experience.")
                
            else:
                st.error("Failed to load video. Please try another talk.")


# ===== RUN APP =====
if __name__ == "__main__":
    main()
