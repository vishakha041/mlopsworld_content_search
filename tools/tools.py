"""
ApertureDB Query Tools for LangGraph Agent

This module contains modular query tools that wrap ApertureDB operations 
for use with LangGraph agents. Each tool handles complete workflows to 
minimize the need for multi-step agent coordination.

Built for the MLOps Events dataset with schema:
- Talk entities with properties like talk_title, speaker_name, yt_views, etc.
- Person entities connected via TalkHasSpeaker edges
- Three descriptor sets: ds_transcript_chunks_v1, ds_talk_meta_v1, ds_speaker_bio_v1
"""

# Import utilities and configurations
from utils import *

# ===== IMPORT TOOL IMPLEMENTATIONS =====
# Import the individual tool implementations
from search_talks_by_filters import search_talks_by_filters
from search_talks_semantically import search_talks_semantically
from analyze_speaker_activity import analyze_speaker_activity
from get_talk_details import get_talk_details
from find_similar_content import find_similar_content
from analyze_topics_and_trends import analyze_topics_and_trends

# Export all tools for easy importing
__all__ = [
    "search_talks_by_filters",
    "search_talks_semantically", 
    "analyze_speaker_activity",
    "get_talk_details",
    "find_similar_content",
    "analyze_topics_and_trends",
    # Utility functions (from utils)
    "get_embedding_model",
    "get_db_connector",
    "to_blob",
    "safe_get",
    "format_date_constraint",
    "get_sort_key",
    "get_sort_description",
    "get_text_field_name",
    # Configuration constants
    "SET_TRANSCRIPT",
    "SET_META", 
    "SET_BIO",
    "EMBED_MODEL",
    "EMBED_DIM"
]



