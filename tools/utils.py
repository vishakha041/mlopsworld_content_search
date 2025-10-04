"""
Shared utilities for ApertureDB Query Tools

This module contains common helper functions and configuration
used across all LangGraph tools for ApertureDB operations.
"""

import os
import numpy as np
from typing import Optional, Any
from datetime import datetime

from load_toml import load_toml_env

# Load environment variables from config.toml (if present)
load_toml_env()

# ApertureDB imports
from aperturedb.CommonLibrary import create_connector
from aperturedb import Connector

# Sentence Transformers for embeddings (matching notebook pattern)
from sentence_transformers import SentenceTransformer

# ===== CONFIGURATION =====
# These should match your setup - can be overridden via environment variables
EMBED_MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-300m")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
APERTUREDB_KEY = os.getenv("APERTUREDB_KEY")

# Descriptor set names (matching your schema)
SET_TRANSCRIPT = "ds_transcript_chunks_v1"
SET_META = "ds_talk_meta_v1" 
SET_BIO = "ds_speaker_bio_v1"

# Global variables for lazy initialization
_embedding_model = None
_db_connector = None

def get_embedding_model():
    """
    Lazy initialization of the embedding model.
    
    For Streamlit apps, this will use session state if available,
    otherwise falls back to global variable.
    """
    global _embedding_model
    
    # Try to use Streamlit session state if available
    try:
        import streamlit as st
        if hasattr(st, 'session_state'):
            if "embedding_model" not in st.session_state or st.session_state.embedding_model is None:
                st.session_state.embedding_model = SentenceTransformer(EMBED_MODEL)
                st.session_state.embedding_model.max_seq_length = 512
            return st.session_state.embedding_model
    except (ImportError, RuntimeError):
        # Not in Streamlit context, use global variable
        pass
    
    # Fallback to global variable for CLI usage
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL)
        _embedding_model.max_seq_length = 512
    return _embedding_model

def get_db_connector():
    """
    Lazy initialization of the ApertureDB connector with retry logic.
    
    For Streamlit apps, this uses session state to maintain a persistent
    connection throughout the session. For CLI usage, it uses a global variable.
    
    Includes automatic retry on connection failure.
    """
    global _db_connector
    
    # Try to use Streamlit session state if available
    try:
        import streamlit as st
        if hasattr(st, 'session_state'):
            # Check if we have a valid connection in session state
            if "db_connector" not in st.session_state or st.session_state.db_connector is None:
                st.session_state.db_connector = _create_connection()
            
            # Test connection and retry if needed
            try:
                # Quick connection test (you can adjust this based on ApertureDB's API)
                _ = st.session_state.db_connector
                return st.session_state.db_connector
            except Exception as e:
                print(f"⚠️ Connection test failed, retrying: {e}")
                st.session_state.db_connector = _create_connection()
                return st.session_state.db_connector
    except (ImportError, RuntimeError):
        # Not in Streamlit context, use global variable
        pass
    
    # Fallback to global variable for CLI usage
    if _db_connector is None:
        _db_connector = _create_connection()
    return _db_connector

def _create_connection():
    """
    Helper function to create a new ApertureDB connection.
    
    Returns:
        Connector: ApertureDB connector instance
    """
    # Re-check environment variable at runtime (in case it was set after import)
    adb_key = os.getenv("APERTUREDB_KEY") or APERTUREDB_KEY
    if not adb_key:
        raise ValueError(
            "APERTUREDB_KEY environment variable must be set. "
            "Please check your config.toml file."
        )
    print(f"🔌 Connecting to ApertureDB (key length: {len(adb_key)} chars)...")
    connector = create_connector(key=adb_key)
    print("✅ ApertureDB connection established!")
    return connector

def to_blob(vec: np.ndarray) -> bytes:
    """
    Convert vector to ApertureDB blob format (float32 little-endian bytes).
    Matches the pattern used in the notebook.
    """
    return np.asarray(vec, dtype="<f4").tobytes()

def safe_get(obj: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary, handling None/missing keys."""
    if obj is None:
        return default
    return obj.get(key, default)

def format_date_constraint(date_str: str) -> Optional[dict]:
    """Convert date string to ApertureDB date format."""
    if not date_str:
        return None
    # Ensure YYYY-MM-DD format
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return {"_date": dt.strftime("%Y-%m-%d")}
    except ValueError:
        # Try to parse other common formats
        for fmt in ["%Y-%m", "%Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                if fmt == "%Y":
                    return {"_date": f"{date_str}-01-01"}
                else:
                    return {"_date": f"{date_str}-01"}
            except ValueError:
                continue
        # If all parsing fails, return None
        return None

def get_sort_key(sort_by: str) -> str:
    """Get ApertureDB sort key from sort_by parameter."""
    sort_mapping = {
        "date": "yt_published_at",
        "views": "yt_views", 
        "title": "talk_title",
        "tech_level": "tech_level"
    }
    return sort_mapping.get(sort_by, "yt_published_at")

def get_sort_description(sort_by: str, sort_order: str) -> str:
    """Get human-readable sort description."""
    if sort_order == "desc":
        if sort_by == "date":
            return "newest"
        elif sort_by == "views":
            return "highest"
        else:
            return "highest"
    else:
        if sort_by == "date":
            return "oldest"
        elif sort_by == "views":
            return "lowest"
        else:
            return "lowest"

def get_text_field_name(set_name: str) -> str:
    """Get the text field name for a descriptor set."""
    if set_name == SET_TRANSCRIPT:
        return "chunk_text"
    elif set_name == SET_META:
        return "meta_text"
    elif set_name == SET_BIO:
        return "bio_text"
    else:
        return "chunk_text"