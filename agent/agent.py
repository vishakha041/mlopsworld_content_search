"""
Minimal LangGraph ReAct Agent for MLOps Events Database

This module implements a simple ReAct agent using LangGraph's prebuilt
create_react_agent function. The agent can query and analyze the MLOps
Events dataset stored in ApertureDB using 6 comprehensive tools.

Key Features:
- Uses Gemini 2.5 Pro for reasoning
- No conversation memory (stateless)
- Comprehensive debugging output
- Simple, beginner-friendly implementation
"""

from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Import all tools from the tools module
from tools.tools import (
    search_talks_by_filters,
    search_talks_semantically,
    analyze_speaker_activity,
    get_talk_details,
    find_similar_content,
    analyze_topics_and_trends,
    get_unique_values
)

# Import system prompt
from agent.prompt import get_system_prompt

# Import configuration
from agent.config import (
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MAX_ITERATIONS,
    GOOGLE_API_KEY
)


def create_mlops_agent():
    """
    Create a ReAct agent for querying MLOps events database.
    
    This function creates a simple agent using LangGraph's prebuilt
    create_react_agent. The agent automatically:
    - Decides which tool(s) to use based on user query
    - Executes the selected tool(s)
    - Reasons about the results
    - Provides a natural language response
    
    Returns:
        CompiledGraph: Compiled LangGraph agent ready for invocation
    """
    # 1. Initialize the Gemini 2.5 Pro model
    print("🔧 Initializing Gemini 2.5 Pro model...")
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    
    # 2. Collect all available tools
    tools = [
        search_talks_by_filters,
        search_talks_semantically,
        analyze_speaker_activity,
        get_talk_details,
        find_similar_content,
        analyze_topics_and_trends,
        get_unique_values
    ]
    
    print(f"🛠️  Loaded {len(tools)} tools:")
    for tool in tools:
        tname = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        tdesc = getattr(tool, "description", getattr(tool, "__doc__", "")) or ""
        print(f"   - {tname}: {tdesc[:80]}...")

    # 3. Get system prompt
    system_prompt = get_system_prompt()
    print(f"📝 System prompt loaded ({len(system_prompt)} characters)")
    
    # 4. Create agent using prebuilt function
    print("🤖 Creating LangGraph ReAct agent...")
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,  # System instructions
    )
    
    print("✅ Agent created successfully!\n")
    return agent


def query_agent(
    user_query: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Query the agent with a user question.
    
    This function provides comprehensive debugging output showing:
    - User query
    - Agent's reasoning steps
    - Tool calls and their arguments
    - Tool results
    - Final response
    
    Args:
        user_query: The user's question
        verbose: If True, print detailed execution steps
        
    Returns:
        Dict containing the full agent response including messages
    """
    # Create agent
    agent = create_mlops_agent()
    
    if verbose:
        print("=" * 80)
        print("🙋 USER QUERY")
        print("=" * 80)
        print(f"{user_query}\n")
        print("=" * 80)
        print("🤖 AGENT EXECUTION")
        print("=" * 80)
    
    # Prepare input
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    
    # Stream the agent's execution to see each step
    step_count = 0
    final_response = None
    
    try:
        for event in agent.stream(inputs, stream_mode="values"):
            step_count += 1
            
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"📍 STEP {step_count}")
                print(f"{'─' * 80}")
            
            # Get the last message in the current state
            messages = event.get("messages", [])
            if not messages:
                continue
                
            last_message = messages[-1]
            
            # Handle different message types
            if hasattr(last_message, 'content'):
                # AI Message or Human Message
                if verbose:
                    print(f"💬 Message Type: {type(last_message).__name__}")
                    
                # Check for tool calls
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"\n🔧 TOOL CALLS ({len(last_message.tool_calls)}):")
                    for i, tool_call in enumerate(last_message.tool_calls, 1):
                        print(f"\n   Tool #{i}: {tool_call['name']}")
                        print(f"   Arguments:")
                        for key, value in tool_call['args'].items():
                            # Truncate long values for readability
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            print(f"      {key}: {value_str}")
                        print(f"   Call ID: {tool_call['id']}")
                
                # Check for regular content
                if last_message.content:
                    content = last_message.content
                    if verbose and len(content) > 0:
                        print(f"\n💭 Content:")
                        # For tool results, format nicely
                        if "ToolMessage" in str(type(last_message)):
                            # Truncate long tool results
                            if len(content) > 500:
                                print(f"   {content[:500]}...")
                                print(f"   ... (truncated, total length: {len(content)} chars)")
                            else:
                                print(f"   {content}")
                        else:
                            print(f"   {content}")
            
            # Store final response
            final_response = event
    
    except Exception as e:
        print(f"\n❌ ERROR during agent execution: {e}")
        raise
    
    if verbose:
        print(f"\n{'=' * 80}")
        print("✅ FINAL RESPONSE")
        print("=" * 80)
        if final_response and "messages" in final_response:
            last_msg = final_response["messages"][-1]
            if hasattr(last_msg, 'content'):
                print(f"{last_msg.content}")
        print("=" * 80)
        print(f"\n📊 Total Steps: {step_count}")
        print("=" * 80)
    
    return final_response


def get_final_answer(response: Dict[str, Any]) -> str:
    """
    Extract the final answer from the agent's response.
    
    Args:
        response: The full agent response dictionary
        
    Returns:
        str: The final answer text
    """
    if response and "messages" in response:
        last_message = response["messages"][-1]
        if hasattr(last_message, 'content'):
            return last_message.content
    return "No response generated."


# Optional: Quick test function
def test_agent():
    """Quick test of the agent with a simple query."""
    test_query = "Give the youtube URLs of talks on fine tuning in Mlops 2024 event."
    print("🧪 Testing agent with query:", test_query)
    response = query_agent(test_query)
    return response


if __name__ == "__main__":
    # Run a test when module is executed directly
    test_agent()
