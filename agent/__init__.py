"""
MLOps Events LangGraph Agent Package

This package contains the LangGraph ReAct agent implementation
for querying the MLOps Events database stored in ApertureDB.
"""

from dotenv import load_dotenv

# Load environment variables first, before any other imports
load_dotenv()

from .agent import create_mlops_agent, query_agent, get_final_answer

__all__ = [
    "create_mlops_agent",
    "query_agent", 
    "get_final_answer",
]
