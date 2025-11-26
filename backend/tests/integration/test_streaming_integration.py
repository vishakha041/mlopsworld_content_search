"""
Integration Tests - Streaming Agent Queries

Tests streaming endpoint with real agent and SSE event format.
"""

import pytest
import json
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_streaming_endpoint_returns_sse_events(integration_client: TestClient, auth_headers: dict):
    """
    Test streaming endpoint returns Server-Sent Events format.

    Verifies:
    - Response has text/event-stream content type
    - Events are in SSE format (data: {json}\\n\\n)
    - Events contain valid JSON with event_type, data, timestamp
    """
    with integration_client.stream(
        "POST",
        "/api/v1/query/stream",
        headers=auth_headers,
        json={"query": "Show me 2 talks from 2024"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read and parse events
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                event_json = line[6:]  # Remove "data: " prefix
                event_data = json.loads(event_json)
                events.append(event_data)

                # Validate event structure
                assert "event_type" in event_data
                assert "data" in event_data
                assert "timestamp" in event_data
                assert event_data["event_type"] in ["step", "answer", "error"]

        # Should have received at least one event
        assert len(events) > 0


@pytest.mark.slow
def test_streaming_endpoint_receives_final_answer(integration_client: TestClient, auth_headers: dict):
    """
    Test streaming endpoint emits final answer event.

    Verifies:
    - Final event has event_type "answer"
    - Answer data contains non-empty answer text
    - Answer includes query and total_steps
    """
    with integration_client.stream(
        "POST",
        "/api/v1/query/stream",
        headers=auth_headers,
        json={"query": "How many talks are in the database?"}
    ) as response:
        assert response.status_code == 200

        # Collect all events
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                event_json = line[6:]
                event_data = json.loads(event_json)
                events.append(event_data)

        # Find answer event
        answer_events = [e for e in events if e["event_type"] == "answer"]
        assert len(answer_events) == 1

        answer_event = answer_events[0]
        assert "answer" in answer_event["data"]
        assert len(answer_event["data"]["answer"]) > 0
        assert "query" in answer_event["data"]
        assert "total_steps" in answer_event["data"]
        assert answer_event["data"]["total_steps"] > 0
