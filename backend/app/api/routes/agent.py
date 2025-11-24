"""
Agent Query Endpoint

Primary endpoint for natural language queries using the LangGraph agent.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.models.requests import AgentQueryRequest
from app.api.models.responses import AgentQueryResponse, AgentStep, StreamEvent
from app.dependencies import get_agent, verify_api_key_skip_options
from datetime import datetime

router = APIRouter()


@router.post("/query", response_model=AgentQueryResponse, dependencies=[Depends(verify_api_key_skip_options)])
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


@router.post("/query/stream", dependencies=[Depends(verify_api_key_skip_options)])
async def stream_agent_query(
    request: AgentQueryRequest,
    agent = Depends(get_agent)
):
    """
    Execute a natural language query with streaming response.

    Returns Server-Sent Events (SSE) with real-time execution steps
    and final answer as they occur.

    **Authentication Required**: Include `X-API-Key` header

    **Response Format**: Server-Sent Events (SSE)
    - Each event: `data: {json}\\n\\n`
    - Event types: "step" (execution steps), "answer" (final result), "error"

    Args:
        request: AgentQueryRequest with user query

    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator():
        try:
            # Import agent functions
            from app.agent.agent import create_mlops_agent, get_final_answer

            # Create agent
            agent_instance = create_mlops_agent()

            # Prepare input
            inputs = {"messages": [{"role": "user", "content": request.query}]}

            # Stream agent execution
            step_count = 0
            final_response = None

            async for event in agent_instance.astream(inputs, stream_mode="values"):
                step_count += 1
                messages = event.get("messages", [])

                if not messages:
                    continue

                last_message = messages[-1]

                # Format step event
                step_data = {
                    "step_number": step_count,
                    "message_type": type(last_message).__name__
                }

                # Extract tool calls if present
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    step_data["tool_calls"] = [
                        {
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                            "id": tc.get("id")
                        }
                        for tc in last_message.tool_calls
                    ]

                # Extract content if present
                if hasattr(last_message, 'content') and last_message.content:
                    content = str(last_message.content)
                    # No truncation for frontend to receive full data
                    step_data["content"] = content

                # Emit step event
                stream_event = StreamEvent(
                    event_type="step",
                    data=step_data,
                    timestamp=datetime.now().isoformat()
                )
                yield f"data: {stream_event.model_dump_json()}\n\n"

                # Store final response
                final_response = event

            # Emit final answer event
            final_answer = get_final_answer(final_response)
            answer_event = StreamEvent(
                event_type="answer",
                data={
                    "answer": final_answer,
                    "total_steps": step_count,
                    "query": request.query
                },
                timestamp=datetime.now().isoformat()
            )
            yield f"data: {answer_event.model_dump_json()}\n\n"

        except Exception as e:
            # Emit error event
            error_event = StreamEvent(
                event_type="error",
                data={"error": str(e)},
                timestamp=datetime.now().isoformat()
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )
