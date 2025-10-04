"""
Custom CSS Styling for Streamlit UI

This module contains custom CSS to enhance the visual appearance
of the Streamlit app.
"""


def get_custom_css():
    """
    Returns custom CSS styling for the Streamlit app.
    
    Returns:
        str: CSS styling as a string
    """
    return """
    <style>
        /* Main container */
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        
        /* Example query buttons */
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        /* Agent steps styling */
        .agent-step {
            padding: 0.5rem;
            margin: 0.3rem 0;
            border-left: 3px solid #ddd;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        
        .agent-step.running {
            border-left-color: #ffc107;
            background-color: #fff9e6;
        }
        
        .agent-step.complete {
            border-left-color: #28a745;
            background-color: #e6f9f0;
        }
        
        .agent-step.error {
            border-left-color: #dc3545;
            background-color: #ffe6e6;
        }
        
        /* Step details */
        .step-number {
            font-weight: bold;
            color: #666;
        }
        
        .step-tool {
            color: #0066cc;
            font-family: monospace;
        }
        
        .step-args {
            font-size: 0.85rem;
            color: #555;
            margin-left: 1rem;
        }
        
        /* Chat message enhancements */
        .user-message {
            background-color: #e3f2fd;
        }
        
        /* Code blocks in responses */
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
        }
    </style>
    """
