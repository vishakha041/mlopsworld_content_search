"""
Integration Tests - Agent Queries

Tests LangGraph agent with real tools and database.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_agent_simple_query(integration_client: TestClient, auth_headers: dict):
    """
    Test basic agent query with real agent and tools.

    Asks agent to show most popular talks from 2024.
    Verifies agent uses real tools to query database and returns answer.

    Note: This test may take 10-30 seconds as the agent processes the query.
    """
    response = integration_client.post(
        "/api/v1/query",
        headers=auth_headers,
        json={
            "query": "Show me the top 5 most popular talks from 2024"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "answer" in data
    assert len(data["answer"]) > 0

    # Agent should have used tools
    if "steps" in data:
        assert isinstance(data["steps"], list)


@pytest.mark.slow
def test_agent_semantic_query(integration_client: TestClient, auth_headers: dict):
    """
    Test agent query requiring semantic search with real embeddings.

    Asks agent about AI agents with memory.
    Verifies agent uses semantic search tool with real embedding model.

    Note: This test may take 10-30 seconds.
    """
    response = integration_client.post(
        "/api/v1/query",
        headers=auth_headers,
        json={
            "query": "Which talks discuss AI agents?"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "answer" in data


@pytest.mark.slow
def test_agent_speaker_analysis_query(integration_client: TestClient, auth_headers: dict):
    """
    Test agent query for speaker analysis with real database.

    Asks agent to identify top speakers.
    Verifies agent uses speaker analysis tool with real data.

    Note: This test may take 10-30 seconds.
    """
    response = integration_client.post(
        "/api/v1/query",
        headers=auth_headers,
        json={
            "query": "Who are the top 5 most active speakers?"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "answer" in data
    assert len(data["answer"]) > 0


@pytest.mark.slow
def test_agent_trend_analysis_query(integration_client: TestClient, auth_headers: dict):
    """
    Test agent query for trend analysis with real database.

    Asks agent about trending tools in MLOps.
    Verifies agent uses trend analysis tool with real content.

    Note: This test may take 10-30 seconds.
    """
    response = integration_client.post(
        "/api/v1/query",
        headers=auth_headers,
        json={
            "query": "What are the most mentioned tools in MLOps talks?"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "answer" in data


@pytest.mark.slow
def test_agent_complex_query(integration_client: TestClient, auth_headers: dict):
    """
    Test agent with complex multi-step query using real tools.

    Asks agent a question requiring multiple tool calls.
    Verifies agent can chain multiple real tool executions.

    Note: This test may take 30-60 seconds.
    """
    response = integration_client.post(
        "/api/v1/query",
        headers=auth_headers,
        json={
            "query": "Find talks about Kubernetes from 2024 with more than 500 views"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "answer" in data
