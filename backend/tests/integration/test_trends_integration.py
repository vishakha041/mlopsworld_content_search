"""
Integration Tests - Trend Analysis

Tests trend analysis and unique values with real database.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_analyze_tools_trend(integration_client: TestClient, auth_headers: dict):
    """
    Test tools trend analysis with real database.

    Extracts and counts tool mentions from actual transcripts and abstracts.
    Verifies tool frequency analysis works correctly.
    """
    response = integration_client.post(
        "/api/v1/trends/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "tools",
            "content_source": "all",
            "top_n": 20,
            "min_mentions": 2
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "analysis_results" in data
    assert "total_items_found" in data

    # Verify tool mentions have proper structure
    if data["total_items_found"] > 0:
        tool = data["analysis_results"][0]
        assert "item" in tool
        assert "count" in tool
        assert tool["count"] >= 2  # min_mentions


def test_analyze_topics_trend(integration_client: TestClient, auth_headers: dict):
    """
    Test topics trend analysis with real database.

    Analyzes topic frequency in transcripts.
    Verifies topic extraction and counting works.
    """
    response = integration_client.post(
        "/api/v1/trends/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "topics",
            "content_source": "transcripts",
            "top_n": 15,
            "min_mentions": 3
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # Verify mentions threshold
    for item in data["analysis_results"]:
        assert item["count"] >= 3


def test_analyze_technologies_trend(integration_client: TestClient, auth_headers: dict):
    """
    Test technologies trend analysis with real database.

    Extracts technology mentions from all content.
    Verifies technology frequency analysis.
    """
    response = integration_client.post(
        "/api/v1/trends/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "technologies",
            "content_source": "all",
            "top_n": 20,
            "min_mentions": 2
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True


def test_analyze_with_time_grouping(integration_client: TestClient, auth_headers: dict):
    """
    Test trend analysis with time grouping on real database.

    Groups tool mentions by month.
    Verifies temporal trend analysis works.
    """
    response = integration_client.post(
        "/api/v1/trends/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "tools",
            "content_source": "all",
            "time_grouping": "monthly",
            "top_n": 15,
            "min_mentions": 2
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # time_trends may be present if grouping is applied
    if data.get("time_trends"):
        assert isinstance(data["time_trends"], dict)


def test_analyze_with_filters(integration_client: TestClient, auth_headers: dict):
    """
    Test trend analysis with category and date filters on real database.

    Analyzes keywords in Deployment category from 2024.
    Verifies combined filtering works.
    """
    response = integration_client.post(
        "/api/v1/trends/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "keywords",
            "content_source": "abstracts",
            "date_from": "2024-01-01",
            "category": "Deployment and integration",
            "top_n": 30,
            "min_mentions": 5
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True


def test_get_all_unique_values(integration_client: TestClient, auth_headers: dict):
    """
    Test retrieving all unique metadata values from real database.

    Gets unique values for all talk properties.
    Verifies complete metadata extraction.
    """
    response = integration_client.post(
        "/api/v1/trends/unique-values",
        headers=auth_headers,
        json={
            "event_name": True,
            "category_primary": True,
            "track": True,
            "company_name": True,
            "tech_level": True,
            "industries": True
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "unique_values" in data
    assert "counts" in data
    assert "total_entities" in data

    # Verify all requested properties are present
    assert "event_name" in data["unique_values"]
    assert "category_primary" in data["unique_values"]
    assert "track" in data["unique_values"]


def test_get_specific_unique_values(integration_client: TestClient, auth_headers: dict):
    """
    Test retrieving specific unique values from real database.

    Gets only event names and categories.
    Verifies selective property retrieval.
    """
    response = integration_client.post(
        "/api/v1/trends/unique-values",
        headers=auth_headers,
        json={
            "event_name": True,
            "category_primary": True
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "event_name" in data["unique_values"]
    assert "category_primary" in data["unique_values"]

    # Verify values are lists
    assert isinstance(data["unique_values"]["event_name"], list)
    assert isinstance(data["unique_values"]["category_primary"], list)


def test_get_company_list(integration_client: TestClient, auth_headers: dict):
    """
    Test retrieving unique company names from real database.

    Gets list of all companies with talks.
    Verifies company name extraction.
    """
    response = integration_client.post(
        "/api/v1/trends/unique-values",
        headers=auth_headers,
        json={
            "company_name": True
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "company_name" in data["unique_values"]
    assert len(data["unique_values"]["company_name"]) > 0
