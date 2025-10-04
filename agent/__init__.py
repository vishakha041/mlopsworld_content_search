"""
MLOps Events LangGraph Agent Package

This package contains the LangGraph ReAct agent implementation
for querying the MLOps Events database stored in ApertureDB.
"""

from load_toml import load_toml_env

# Load environment variables first, before any other imports
load_toml_env()

from .agent import create_mlops_agent, query_agent, get_final_answer

__all__ = [
    "create_mlops_agent",
    "query_agent", 
    "get_final_answer",
]
