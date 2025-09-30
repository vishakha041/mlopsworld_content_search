"""
Simple CLI Interface for MLOps Events LangGraph Agent

This script provides a command-line interface to interact with the
LangGraph agent. You can ask questions about MLOps conference talks
and get intelligent responses powered by Gemini 2.5 Pro and ApertureDB.

Usage:
    python app.py

Features:
- Interactive chat loop
- Comprehensive debugging output
- Easy to test agent behavior
- Graceful error handling
"""

import sys
from agent.agent import query_agent, get_final_answer


def print_banner():
    """Print a nice welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🤖 MLOps Events Agent (LangGraph + ApertureDB)                    ║
║                                                                              ║
║  Powered by: Gemini 2.5 Pro + LangGraph + ApertureDB                       ║
║  Dataset: 278 MLOps & GenAI conference talks                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ask questions about MLOps conference talks, speakers, trends, and more!

Example queries:
  • "What are the top 10 most viewed talks?"
  • "Find talks about AI agents with memory"
  • "Show me talks from 2024 about deployment"
  • "Who are the most active speakers?"
  • "What tools are trending in MLOps?"

Commands:
  • Type 'quit', 'exit', or 'q' to exit
  • Type 'help' for more example queries
  • Type 'clear' to clear the screen

"""
    print(banner)


def print_help():
    """Print help with example queries."""
    help_text = """
📚 EXAMPLE QUERIES:

FILTERING & SORTING:
  • "Show me the most popular talks from 2024"
  • "Find talks by speakers from Google"
  • "What are the most viewed MLOps talks?"

SEMANTIC SEARCH:
  • "Which talks discuss AI agents and memory?"
  • "Find experts in vector databases"
  • "Talks about RAG deployment strategies"

SPEAKER ANALYSIS:
  • "How many times has John Smith presented?"
  • "Who are the top 10 speakers?"
  • "Which companies presented the most?"

DETAILED INFORMATION:
  • "Tell me about the LangChain production talk"
  • "Show me the transcript from 5-10 minutes of this talk"
  • "Get details about the RAG deployment presentation"

RECOMMENDATIONS:
  • "Find talks similar to the multimodal agents presentation"
  • "Recommend content about deployment strategies"
  • "Show me related talks to this one"

TRENDS & ANALYSIS:
  • "What are the top 10 tools mentioned in talks?"
  • "Show trending technologies in 2024"
  • "What topics are most discussed?"

"""
    print(help_text)


def main():
    """Main CLI loop."""
    print_banner()
    
    # Check if running in verbose mode
    verbose = "--quiet" not in sys.argv
    
    print("💡 Tip: The agent will show detailed execution steps for debugging.\n")
    print("Ready! Ask your first question:\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🙋 You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using MLOps Events Agent. Goodbye!\n")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            
            # Query the agent
            print()  # Empty line for readability
            response = query_agent(user_input, verbose=verbose)
            
            # Print a separator for next query
            print("\n" + "─" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()
