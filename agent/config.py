"""
Agent Configuration for MLOps Events LangGraph Agent

This module contains configuration settings for the LangGraph agent,
including model settings and agent parameters.
"""

import os
from load_toml import load_toml_env

# Load environment variables from config.toml (if present)
load_toml_env()

# ===== MODEL CONFIGURATION =====
MODEL_PROVIDER = "google-genai"
MODEL_NAME = "gemini-2.5-pro"
MODEL_TEMPERATURE = 0.7

# ===== API KEYS =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
APERTUREDB_KEY = os.getenv("APERTUREDB_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please set it in your config.toml file."
    )

if not APERTUREDB_KEY:
    raise ValueError(
        "APERTUREDB_KEY not found in environment variables. "
        "Please set it in your config.toml file."
    )

# ===== AGENT CONFIGURATION =====
MAX_ITERATIONS = 10  # Maximum number of agent reasoning steps
ENABLE_VERBOSE_OUTPUT = True  # Enable detailed logging for debugging

# ===== TOOL CONFIGURATION =====
# These are already configured in tools/utils.py
# Just documenting here for clarity
EMBED_MODEL = "google/embeddinggemma-300m"
EMBED_DIM = 768
