"""
Agent Configuration for MLOps Events LangGraph Agent

This module imports configuration from the main app config.
"""

from app.config import settings

# ===== MODEL CONFIGURATION =====
MODEL_PROVIDER = settings.MODEL_PROVIDER
MODEL_NAME = settings.MODEL_NAME
MODEL_TEMPERATURE = settings.MODEL_TEMPERATURE

# ===== API KEYS =====
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
APERTUREDB_KEY = settings.APERTUREDB_KEY

# ===== AGENT CONFIGURATION =====
MAX_ITERATIONS = settings.MAX_ITERATIONS
ENABLE_VERBOSE_OUTPUT = settings.ENABLE_VERBOSE_OUTPUT

# ===== TOOL CONFIGURATION =====
EMBED_MODEL = settings.EMBED_MODEL
EMBED_DIM = settings.EMBED_DIM
