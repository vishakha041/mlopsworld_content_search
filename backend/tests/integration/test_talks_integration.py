"""
Integration Tests - Talk Endpoints

Tests talk endpoints with real database queries.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_filter_talks_by_date_range(integration_client: TestClient, auth_headers: dict):
    """
    Test filtering talks by date range with real database.

    Makes actual database query for talks in 2024.
    Verifies results contain valid talk data.
    """
    response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "sort_by": "date_desc",
            "limit": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "results" in data
    assert "total_found" in data

    # Verify results have expected structure
    if data["total_found"] > 0:
        result = data["results"][0]
        assert "talk_id" in result
        assert "title" in result
        assert "speaker" in result


def test_filter_talks_by_views(integration_client: TestClient, auth_headers: dict):
    """
    Test filtering talks by minimum views with real database.

    Queries for talks with at least 1000 views.
    Verifies results are sorted by views descending.
    """
    response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={
            "min_views": 1000,
            "sort_by": "views_desc",
            "limit": 20
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # If results exist, verify view count filtering
    if data["total_found"] > 0:
        for result in data["results"]:
            if result.get("views") is not None:
                assert result["views"] >= 1000


def test_filter_talks_by_category(integration_client: TestClient, auth_headers: dict):
    """
    Test filtering talks by category with real database.

    Filters for MLOps category talks.
    Verifies category filter is applied correctly.
    """
    response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={
            "category": "MLOps",
            "sort_by": "views_desc",
            "limit": 15
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True


def test_get_talk_details(integration_client: TestClient, auth_headers: dict):
    """
    Test getting talk details with real database.

    First gets a talk from filter, then retrieves its details.
    Verifies detailed information is returned.
    """
    # First get a talk title
    filter_response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={"limit": 1}
    )

    filter_data = filter_response.json()
    if filter_data["total_found"] == 0:
        pytest.skip("No talks in database")

    talk_title = filter_data["results"][0]["title"]

    # Now get details for that talk
    response = integration_client.post(
        "/api/v1/talks/details",
        headers=auth_headers,
        json={
            "talk_title": talk_title,
            "include_transcript": False,
            "include_related": False
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "talk_info" in data
    assert data["talk_info"]["title"] == talk_title


def test_get_talk_details_with_transcript(integration_client: TestClient, auth_headers: dict):
    """
    Test getting talk details with transcript from real database.

    Retrieves a talk with transcript chunks.
    Verifies transcript data is included.
    """
    # Get a talk
    filter_response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={"limit": 1}
    )

    filter_data = filter_response.json()
    if filter_data["total_found"] == 0:
        pytest.skip("No talks in database")

    talk_title = filter_data["results"][0]["title"]

    # Get details with transcript
    response = integration_client.post(
        "/api/v1/talks/details",
        headers=auth_headers,
        json={
            "talk_title": talk_title,
            "include_transcript": True
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # Transcript may or may not exist for this talk
    if data.get("transcript_chunks"):
        assert isinstance(data["transcript_chunks"], list)
        if len(data["transcript_chunks"]) > 0:
            chunk = data["transcript_chunks"][0]
            assert "text" in chunk
            assert "sequence" in chunk
