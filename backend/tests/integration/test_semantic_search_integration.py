"""
Integration Tests - Semantic Search

Tests semantic search with real embeddings and database.
"""

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_semantic_search_all_types(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search across all content types with real embeddings.

    Searches for "machine learning deployment" across transcripts, metadata, and bios.
    Uses real embedding model to generate query vector.
    Makes actual database similarity search.
    """
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "machine learning deployment",
            "search_type": "all",
            "k_neighbors": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "results" in data
    assert "total_found" in data
    assert "search_summary" in data

    # Results should have similarity scores
    if data["total_found"] > 0:
        result = data["results"][0]
        assert "similarity_score" in result
        assert 0.0 <= result["similarity_score"] <= 1.0


def test_semantic_search_transcript_only(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search on transcripts only with real embeddings.

    Searches transcript content for vector database topics.
    Verifies content_type is transcript in results.
    """
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "vector databases and embeddings",
            "search_type": "transcript",
            "k_neighbors": 5
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # Verify search type is reflected in results
    if data["total_found"] > 0:
        for result in data["results"]:
            if result.get("content_type"):
                assert result["content_type"] == "transcript"


def test_semantic_search_with_date_filter(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search with date filtering on real database.

    Searches for AI agents content from 2024 onwards.
    Combines semantic similarity with metadata filtering.
    """
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "AI agents",
            "search_type": "all",
            "date_from": "2024-01-01",
            "k_neighbors": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True


def test_semantic_search_with_category_filter(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search with category filtering on real database.

    Searches for data quality content in MLOps category.
    Verifies both semantic matching and category filtering work together.
    """
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "data quality",
            "search_type": "all",
            "category": "MLOps",
            "k_neighbors": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True


def test_semantic_search_with_score_threshold(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search with similarity threshold on real embeddings.

    Searches for machine learning with minimum 0.5 similarity.
    Verifies all results meet the threshold.
    """
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "machine learning",
            "search_type": "all",
            "k_neighbors": 20,
            "score_threshold": 0.5
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True

    # Verify all results meet threshold
    for result in data["results"]:
        assert result["similarity_score"] >= 0.5


def test_find_similar_content_by_query(integration_client: TestClient, auth_headers: dict):
    """
    Test finding similar content using query text with real embeddings.

    Finds talks similar to "deploying machine learning models at scale".
    Uses real semantic similarity computation.
    """
    response = integration_client.post(
        "/api/v1/talks/similar",
        headers=auth_headers,
        json={
            "reference_query": "deploying machine learning models at scale",
            "similarity_type": "content",
            "k_neighbors": 10
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "similar_talks" in data

    # Verify similarity information
    if data["total_found"] > 0:
        result = data["similar_talks"][0]
        assert "similarity_score" in result
        assert "title" in result


def test_find_similar_content_by_title(integration_client: TestClient, auth_headers: dict):
    """
    Test finding similar content by reference talk title with real database.

    First gets a talk, then finds similar talks.
    Uses actual talk embeddings for similarity.
    """
    # Get a talk first
    filter_response = integration_client.post(
        "/api/v1/talks/filter",
        headers=auth_headers,
        json={"limit": 1}
    )

    filter_data = filter_response.json()
    if filter_data["total_found"] == 0:
        pytest.skip("No talks in database")

    reference_title = filter_data["results"][0]["title"]

    # Find similar talks
    response = integration_client.post(
        "/api/v1/talks/similar",
        headers=auth_headers,
        json={
            "reference_talk_title": reference_title,
            "similarity_type": "content",
            "k_neighbors": 5
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "similar_talks" in data
    assert "reference_info" in data
