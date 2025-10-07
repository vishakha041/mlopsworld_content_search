"""
Semantic Video Search Page

Search for MLOps conference talk videos using natural language queries.
Uses video embeddings stored in ApertureDB to find semantically similar content.
"""

import streamlit as st
import sys
import os
import numpy as np
import tempfile
import logging
from dotenv import load_dotenv
from twelvelabs import TwelveLabs

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.utils import get_db_connector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Semantic Search - MLOps Events",
    page_icon="🔍",
    layout="wide"
)


# ===== CONFIGURATION =====
DESCRIPTOR_SET = "marengo_2_7"  # Name of the descriptor set in ApertureDB
MODEL_NAME = "Marengo-retrieval-2.7"  # Twelve Labs embedding model
EMBEDDING_DIMENSIONS = 1024


# ===== HELPER FUNCTIONS =====

@st.cache_resource
def get_twelvelabs_client():
    """
    Get or create Twelve Labs client instance.
    Cached to avoid recreating the client on every interaction.
    """
    try:
        api_key = os.getenv('TL_API_KEY')
        if not api_key:
            st.error("TL_API_KEY not found in environment variables")
            return None
        
        tl = TwelveLabs(api_key=api_key)
        logging.info("Twelve Labs client initialized")
        return tl
    except Exception as e:
        st.error(f"Failed to initialize Twelve Labs client: {e}")
        logging.error(f"Failed to initialize Twelve Labs client: {e}")
        return None


# ===== SEARCH FUNCTION =====

def semantic_search_videos(query_text, top_n=3):
    """
    Perform semantic search on videos using text query.
    
    Args:
        query_text (str): Natural language search query
        top_n (int): Number of results to return
        
    Returns:
        list: List of dictionaries containing video metadata and blobs
    """
    try:
        # Get database connector and Twelve Labs client
        client = get_db_connector()
        tl = get_twelvelabs_client()
        
        if not tl:
            st.error("Twelve Labs client not available")
            return []
        
        # 1. Create text embedding for the query using Twelve Labs
        logging.info(f"Creating embedding for query: '{query_text}'")
        t = tl.embed.create(model_name=MODEL_NAME, text=query_text)
        qvec = np.asarray(t.text_embedding.segments[0].float_, dtype=np.float32)
        
        # 2. Perform k-NN search in ApertureDB
        logging.info(f"Searching for top {top_n} similar videos...")
        search_q = [
            {
                "FindDescriptor": {
                    "_ref": 2,
                    "set": DESCRIPTOR_SET,
                    "k_neighbors": top_n,
                    "metric": "L2",
                    "distances": True
                }
            },
            {
                "FindVideo": {
                    "is_connected_to": {"ref": 2},  # Reference the previous command
                    "results": {"all_properties": True},
                    "blobs": False
                }
            }
        ]
        
        search_results, _ = client.query(search_q, blobs=[qvec.tobytes()])
        
        # 3. Check and parse results
        find_video_response = search_results[1]
        if find_video_response.get("FindVideo", {}).get("returned", 0) == 0:
            logging.warning("No videos found matching search query")
            return []
        
        video_entities = search_results[1]["FindVideo"]["entities"]
        descriptor_entities = search_results[0]["FindDescriptor"]["entities"]
        video_ids = [entity["_uniqueid"] for entity in video_entities]
        distances = [entity["_distance"] for entity in descriptor_entities]
        
        # 4. Retrieve the video blobs for display
        logging.info("Retrieving video blobs...")
        get_blobs_q = [{
            "FindVideo": {
                "constraints": {"_uniqueid": ["in", video_ids]},
                "blobs": True,
                "results": {"all_properties": True}
            }
        }]
        
        response, blobs = client.query(get_blobs_q)
        
        if not blobs:
            logging.error("Search returned matches but failed to retrieve video blobs")
            return []
        
        # 5. Map blobs to IDs and create results
        retrieved_videos = response[0]["FindVideo"]["entities"]
        blob_map = {entity["_uniqueid"]: blob for entity, blob in zip(retrieved_videos, blobs)}
        
        results = []
        for i, video_id in enumerate(video_ids):
            entity = video_entities[i]
            blob = blob_map.get(video_id)
            
            if blob:
                results.append({
                    "talk_title": entity.get("talk_title", "N/A"),
                    "speaker_name": entity.get("speaker_name", "Unknown"),
                    "distance": distances[i],
                    "video_blob": blob,
                    "metadata": entity
                })
        
        logging.info(f"Found {len(results)} matching videos")
        return results
        
    except Exception as e:
        logging.error(f"Error during semantic search: {str(e)}")
        st.error(f"Search failed: {str(e)}")
        return []


def display_video(video_blob):
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


# ===== MAIN PAGE UI =====

def main():
    """Main function to render the Semantic Search page."""
    
    # Header
    st.title("🔍 Semantic Video Search")
    st.markdown("""
    Search for conference talk videos using natural language queries. 
    The search uses AI embeddings to find videos that are semantically similar to your query.
    """)
    
    st.markdown("---")
    
    # Search input section
    st.subheader("Search Query")
    
    # Example queries in an expander
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown("""
        **Try these example queries:**
        - "AI agents and LLMs in production"
        - "Vector databases and RAG systems"
        - "MLOps best practices and deployment"
        - "Model monitoring and evaluation"
        - "Data quality and feature engineering"
        - "Generative AI and prompt engineering"
        """)
    
    
    # Initialize session state for search query
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    # Search input
    search_query = st.text_input(
        "Enter your search query:",
        value=st.session_state.search_query,
        placeholder="e.g., 'AI agents with memory and context'",
        key="search_input"
    )
    
    # Number of results slider
    top_n = st.slider(
        "Number of results to show:",
        min_value=1,
        max_value=10,
        value=3,
        help="Select how many matching videos to display"
    )
    
    # Search button
    search_clicked = st.button("🔍 Search Videos", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Perform search when button is clicked
    if search_clicked and search_query.strip():
        # Update session state
        st.session_state.search_query = search_query
        
        # Show searching status
        with st.spinner(f"🔍 Searching for '{search_query}'..."):
            results = semantic_search_videos(search_query, top_n)
        
        # Display results
        if results:
            st.success(f"✅ Found {len(results)} matching videos")
            
            # Display each result
            for i, result in enumerate(results, 1):
                st.markdown(f"### Match #{i}: {result['talk_title']}")
                
                # Create columns for metadata
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Speaker:** {result['speaker_name']}")
                
                with col2:
                    # Calculate similarity score (inverse of distance, normalized)
                    similarity = max(0, 1 - (result['distance'] / 10))  # Rough normalization
                    st.markdown(f"**Similarity:** {similarity:.1%}")
                
                with col3:
                    st.markdown(f"**Distance:** {result['distance']:.4f}")
                
                # Display video metadata if available
                metadata = result.get('metadata', {})
                if metadata:
                    cols = st.columns(4)
                    
                    with cols[0]:
                        fps = metadata.get("_fps", 0)
                        if fps:
                            st.metric("FPS", f"{fps:.2f}")
                    
                    with cols[1]:
                        frame_count = metadata.get("_frame_count", 0)
                        duration = frame_count / fps if fps else 0
                        if duration:
                            st.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s")
                    
                    with cols[2]:
                        height = metadata.get("_frame_height", 0)
                        if height:
                            st.metric("Height", f"{height}p")
                    
                    with cols[3]:
                        width = metadata.get("_frame_width", 0)
                        if width:
                            st.metric("Width", f"{width}px")
                
                # Display video
                st.markdown("#### Video")
                display_video(result['video_blob'])
                
                # Add separator between results
                if i < len(results):
                    st.markdown("---")
        
        else:
            st.warning("No matching videos found. Try a different query or broader search terms.")
    
    elif search_clicked and not search_query.strip():
        st.warning("⚠️ Please enter a search query")
    
    # Footer with helpful tips
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        <p><strong>💡 Tips:</strong></p>
        <p>• Use natural language queries for best results</p>
        <p>• The search understands context and semantic meaning</p>
        <p>• Lower distance values indicate better matches</p>
    </div>
    """, unsafe_allow_html=True)


# ===== RUN APP =====
if __name__ == "__main__":
    main()
