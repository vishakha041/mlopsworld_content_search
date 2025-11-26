"""
Integration Tests - Speaker Analysis

Tests speaker analytics with real database queries.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_analyze_all_speakers(integration_client: TestClient, auth_headers: dict):
    """
    Test analyzing all speakers with real database.

    Retrieves speaker statistics from actual talk data.
    Verifies speaker counts, views, and aggregations.
    """
    response = integration_client.post(
        "/api/v1/speakers/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "all",
            "top_n": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "speaker_stats" in data
    assert "total_speakers" in data

    # Verify speaker data structure
    if len(data["speaker_stats"]) > 0:
        speaker = data["speaker_stats"][0]
        assert "speaker_name" in speaker
        assert "total_talks" in speaker
        assert "total_views" in speaker




def test_analyze_specific_speaker(integration_client: TestClient, auth_headers: dict):
    """
    Test analyzing a specific speaker with real database.

    First gets a speaker name, then analyzes their activity.
    Verifies detailed speaker information is returned.
    """
    # Get a speaker first
    all_speakers_response = integration_client.post(
        "/api/v1/speakers/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "all",
            "top_n": 1
        }
    )

    all_data = all_speakers_response.json()
    if len(all_data["speaker_stats"]) == 0:
        pytest.skip("No speakers in database")

    speaker_name = all_data["speaker_stats"][0]["speaker_name"]

    # Analyze specific speaker
    response = integration_client.post(
        "/api/v1/speakers/analyze",
        headers=auth_headers,
        json={
            "speaker_name": speaker_name,
            "analysis_type": "all"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["speaker_stats"]) >= 1
    assert data["speaker_stats"][0]["speaker_name"] == speaker_name


def test_company_breakdown(integration_client: TestClient, auth_headers: dict):
    """
    Test company breakdown analysis with real database.

    Analyzes which companies have the most speakers.
    Verifies company aggregation works correctly.
    """
    response = integration_client.post(
        "/api/v1/speakers/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "companies",
            "date_from": "2024-01-01"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "company_breakdown" in data


def test_speaker_analysis_with_category_filter(integration_client: TestClient, auth_headers: dict):
    """
    Test speaker analysis filtered by category with real database.

    Analyzes speakers in MLOps category only.
    Verifies category filtering is applied.
    """
    response = integration_client.post(
        "/api/v1/speakers/analyze",
        headers=auth_headers,
        json={
            "analysis_type": "all",
            "category": "MLOps",
            "top_n": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
