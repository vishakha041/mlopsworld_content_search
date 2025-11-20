"""
Integration Tests - Health Endpoints

Tests health and status endpoints with real services.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_health_endpoint(integration_client: TestClient):
    """
    Test basic health endpoint returns healthy status.

    Verifies:
    - Returns 200 OK
    - Status is healthy
    - Version is present
    """
    response = integration_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_detailed_status(integration_client: TestClient, auth_headers: dict):
    """
    Test detailed status endpoint with real services.

    Verifies:
    - Returns 200 OK
    - All services are operational
    - Database connection is working
    - Embedding model is loaded
    - Agent is initialized
    """
    response = integration_client.get("/api/v1/status", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "services" in data

    # Verify all services are reported
    services = data["services"]
    assert "database" in services
    assert "embedding_model" in services
    assert "agent" in services
