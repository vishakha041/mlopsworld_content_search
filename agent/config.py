"""
Agent Configuration for MLOps Events LangGraph Agent

This module contains configuration settings for the LangGraph agent,
including model settings and agent parameters.
"""

import os

# ===== MODEL CONFIGURATION =====
MODEL_PROVIDER = "google-genai"
MODEL_NAME = "gemini-2.5-pro"
MODEL_TEMPERATURE = 0.7

# ===== API KEYS =====
def get_secret(key: str) -> str:
    """
    Get secret from Streamlit secrets or environment variables.
    Tries Streamlit secrets first (for deployment), then falls back to env vars (for local dev).
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except (ImportError, FileNotFoundError):
        # Streamlit not available or secrets.toml not found - use environment variables
        return os.getenv(key, "")

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
APERTUREDB_KEY = get_secret("APERTUREDB_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please set it in .streamlit/secrets.toml or environment variables."
    )

if not APERTUREDB_KEY:
    raise ValueError(
        "APERTUREDB_KEY not found. "
        "Please set it in .streamlit/secrets.toml or environment variables."
    )

# ===== AGENT CONFIGURATION =====
MAX_ITERATIONS = 10  # Maximum number of agent reasoning steps
ENABLE_VERBOSE_OUTPUT = True  # Enable detailed logging for debugging

# ===== TOOL CONFIGURATION =====
# These are already configured in tools/utils.py
# Just documenting here for clarity
EMBED_MODEL = "google/embeddinggemma-300m"
EMBED_DIM = 768
