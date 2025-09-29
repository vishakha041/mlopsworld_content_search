"""
Shared utilities for ApertureDB Query Tools

This module contains common helper functions and configuration
used across all LangGraph tools for ApertureDB operations.
"""

import os
import numpy as np
from typing import Optional, Any
from datetime import datetime

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
    """Lazy initialization of the embedding model to match notebook pattern."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL)
        _embedding_model.max_seq_length = 512
    return _embedding_model

def get_db_connector():
    """Lazy initialization of the ApertureDB connector."""
    global _db_connector
    if _db_connector is None:
        if not APERTUREDB_KEY:
            raise ValueError("APERTUREDB_KEY environment variable must be set")
        _db_connector = create_connector(key=APERTUREDB_KEY)
    return _db_connector

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