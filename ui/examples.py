"""
Example Queries Configuration

This module contains curated example queries organized by category.
"""

EXAMPLE_QUERIES = {
    "filtering": {
        "label": "🔍 Filtering & Sorting",
        "queries": [
            "Show me the most popular talks from 2024",
            "Find talks by speakers from Google",
        ]
    },
    "semantic": {
        "label": "🧠 Semantic Search",
        "queries": [
            "Which talks discuss AI agents with memory?",
            "Find experts in vector databases and RAG",
        ]
    },
    "speaker": {
        "label": "👤 Speaker Analysis",
        "queries": [
            "Who are the top 10 most active speakers?",
            "Which companies presented the most talks?",
        ]
    },
    "trends": {
        "label": "📊 Trends & Tools",
        "queries": [
            "What are the most discussed tools in 2024?",
            "Show trending technologies in MLOps",
        ]
    }
}


def get_all_examples():
    """
    Get all example queries as a flat list.
    
    Returns:
        List of all example query strings
    """
    examples = []
    for category_data in EXAMPLE_QUERIES.values():
        examples.extend(category_data["queries"])
    return examples


def get_examples_by_category(category: str):
    """
    Get example queries for a specific category.
    
    Args:
        category: Category key (filtering, semantic, speaker, trends)
        
    Returns:
        List of query strings or empty list if category not found
    """
    return EXAMPLE_QUERIES.get(category, {}).get("queries", [])
