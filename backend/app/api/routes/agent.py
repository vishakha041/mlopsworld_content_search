"""
Agent Query Endpoint

Primary endpoint for natural language queries using the LangGraph agent.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import AgentQueryRequest
from app.api.models.responses import AgentQueryResponse, AgentStep
from app.dependencies import get_agent, verify_api_key
from datetime import datetime

router = APIRouter()


@router.post("/query", response_model=AgentQueryResponse, dependencies=[Depends(verify_api_key)])
async def query_agent(
    request: AgentQueryRequest,
    agent = Depends(get_agent)
):
    """
    Execute a natural language query using the LangGraph agent.

    The agent will automatically select and execute the appropriate tools
    based on the query, providing a natural language response with the results.

    **Authentication Required**: Include `X-API-Key` header

    **Example Queries**:
    - "Show me the most popular talks from 2024"
    - "Which talks discuss AI agents with memory?"
    - "Who are the top 10 most active speakers?"
    - "Find talks about vector databases and RAG"
    - "What tools are trending in MLOps?"

    Args:
        request: AgentQueryRequest with user query

    Returns:
        AgentQueryResponse with answer, execution steps, and metadata
    """
    try:
        # Import here to avoid circular imports
        from app.agent.agent import query_agent as execute_agent_query, get_final_answer

        # Execute agent query
        response = execute_agent_query(request.query, verbose=True)

        # Extract steps from response
        steps = []
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                # Parse message to extract step information
                if hasattr(msg, "type"):
                    step = AgentStep(
                        type=msg.type,
                        content=str(msg.content) if hasattr(msg, "content") else str(msg),
                        timestamp=datetime.now().isoformat()
                    )
                    steps.append(step)

        # Get final answer
        final_answer = get_final_answer(response)

        return AgentQueryResponse(
            success=True,
            answer=final_answer,
            steps=steps,
            metadata={
                "query": request.query,
                "model": "gemini-2.5-pro",
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent query failed: {str(e)}"
        )
