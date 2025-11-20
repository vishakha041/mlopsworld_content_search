"""
MLOps Events API Client

Client wrapper for communicating with the FastAPI backend.
"""

import os
import requests
from typing import Dict, Any, Optional, List


class MLOpsAPIClient:
    """
    Client for interacting with the MLOps Events FastAPI backend.

    Handles all API communication including:
    - Agent queries
    - Talk searches and filtering
    - Speaker analytics
    - Video search
    - Trend analysis
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
            api_key: API key for authentication (from environment if not provided)
        """
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("API_KEY", "your-secret-key-change-in-production")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data (for POST requests)

        Returns:
            Response JSON as dictionary

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=60)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=60)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            raise

    # ===== Health & Status =====

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request("GET", "/health")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed API status"""
        return self._make_request("GET", "/api/v1/status")

    # ===== Agent Query =====

    def query(self, query_text: str, stream: bool = False) -> Dict[str, Any]:
        """
        Execute a natural language query using the agent.

        Args:
            query_text: Natural language query
            stream: Enable streaming (not yet implemented)

        Returns:
            Agent response with answer and steps
        """
        return self._make_request("POST", "/api/v1/query", {
            "query": query_text,
            "stream": stream
        })

    # ===== Talk Endpoints =====

    def search_talks_semantically(
        self,
        query: str,
        search_type: str = "all",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        category: Optional[str] = None,
        event_name: Optional[str] = None,
        speaker_name: Optional[str] = None,
        k_neighbors: int = 10,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Semantic search across talks.

        Args:
            query: Search query
            search_type: Content type to search (transcript/meta/bio/all)
            date_from: Filter from date
            date_to: Filter to date
            category: Category filter
            event_name: Event filter
            speaker_name: Speaker filter
            k_neighbors: Number of results
            score_threshold: Minimum similarity score

        Returns:
            Search results with similarity scores
        """
        return self._make_request("POST", "/api/v1/talks/search", {
            "query": query,
            "search_type": search_type,
            "date_from": date_from,
            "date_to": date_to,
            "category": category,
            "event_name": event_name,
            "speaker_name": speaker_name,
            "k_neighbors": k_neighbors,
            "score_threshold": score_threshold
        })

    def filter_talks(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_views: Optional[int] = None,
        max_views: Optional[int] = None,
        category: Optional[str] = None,
        event_name: Optional[str] = None,
        track: Optional[str] = None,
        tech_level: Optional[int] = None,
        speaker_name: Optional[str] = None,
        company_name: Optional[str] = None,
        sort_by: str = "date_desc",
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Filter talks by metadata.

        Args:
            date_from: Filter from date
            date_to: Filter to date
            min_views: Minimum views
            max_views: Maximum views
            category: Category filter
            event_name: Event filter
            track: Track filter
            tech_level: Technical level (1-7)
            speaker_name: Speaker filter
            company_name: Company filter
            sort_by: Sort order
            limit: Maximum results

        Returns:
            Filtered talks
        """
        return self._make_request("POST", "/api/v1/talks/filter", {
            "date_from": date_from,
            "date_to": date_to,
            "min_views": min_views,
            "max_views": max_views,
            "category": category,
            "event_name": event_name,
            "track": track,
            "tech_level": tech_level,
            "speaker_name": speaker_name,
            "company_name": company_name,
            "sort_by": sort_by,
            "limit": limit
        })

    def get_talk_details(
        self,
        talk_title: Optional[str] = None,
        talk_id: Optional[str] = None,
        include_transcript: bool = False,
        include_related: bool = False,
        related_count: int = 5,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a talk.

        Args:
            talk_title: Talk title
            talk_id: Talk ID
            include_transcript: Include full transcript
            include_related: Include related talks
            related_count: Number of related talks
            time_start: Transcript start time (seconds)
            time_end: Transcript end time (seconds)

        Returns:
            Detailed talk information
        """
        return self._make_request("POST", "/api/v1/talks/details", {
            "talk_title": talk_title,
            "talk_id": talk_id,
            "include_transcript": include_transcript,
            "include_related": include_related,
            "related_count": related_count,
            "time_start": time_start,
            "time_end": time_end
        })

    def find_similar_talks(
        self,
        reference_talk_title: Optional[str] = None,
        reference_talk_id: Optional[str] = None,
        reference_query: Optional[str] = None,
        similarity_type: str = "content",
        exclude_same_speaker: bool = False,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        category: Optional[str] = None,
        event_name: Optional[str] = None,
        k_neighbors: int = 10,
        min_similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find similar talks.

        Args:
            reference_talk_title: Reference talk title
            reference_talk_id: Reference talk ID
            reference_query: Reference query
            similarity_type: Similarity type (content/speaker/topic/all)
            exclude_same_speaker: Exclude same speaker
            date_from: Filter from date
            date_to: Filter to date
            category: Category filter
            event_name: Event filter
            k_neighbors: Number of results
            min_similarity: Minimum similarity score

        Returns:
            Similar talks
        """
        return self._make_request("POST", "/api/v1/talks/similar", {
            "reference_talk_title": reference_talk_title,
            "reference_talk_id": reference_talk_id,
            "reference_query": reference_query,
            "similarity_type": similarity_type,
            "exclude_same_speaker": exclude_same_speaker,
            "date_from": date_from,
            "date_to": date_to,
            "category": category,
            "event_name": event_name,
            "k_neighbors": k_neighbors,
            "min_similarity": min_similarity
        })

    # ===== Speaker Endpoints =====

    def analyze_speaker_activity(
        self,
        speaker_name: Optional[str] = None,
        analysis_type: str = "all",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        category: Optional[str] = None,
        min_talks: int = 1,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze speaker activity.

        Args:
            speaker_name: Specific speaker to analyze
            analysis_type: Analysis type (talk_count/topics/companies/all)
            date_from: Filter from date
            date_to: Filter to date
            category: Category filter
            min_talks: Minimum talks for inclusion
            top_n: Number of top speakers

        Returns:
            Speaker analytics
        """
        return self._make_request("POST", "/api/v1/speakers/analyze", {
            "speaker_name": speaker_name,
            "analysis_type": analysis_type,
            "date_from": date_from,
            "date_to": date_to,
            "category": category,
            "min_talks": min_talks,
            "top_n": top_n
        })

    # ===== Video Endpoints =====

    def search_videos_semantically(
        self,
        query: str,
        top_n: int = 10,
        include_videos: bool = False
    ) -> Dict[str, Any]:
        """
        Semantic video search.

        Args:
            query: Search query
            top_n: Number of results
            include_videos: Include video blobs

        Returns:
            Video search results
        """
        return self._make_request("POST", "/api/v1/videos/search", {
            "query": query,
            "top_n": top_n,
            "include_videos": include_videos
        })

    # ===== Trend Endpoints =====

    def analyze_trends(
        self,
        analysis_type: str,
        content_source: str = "all",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        category: Optional[str] = None,
        event_name: Optional[str] = None,
        time_grouping: Optional[str] = None,
        top_n: int = 20,
        min_mentions: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze trends and topics.

        Args:
            analysis_type: Analysis type (tools/topics/technologies/trends/keywords)
            content_source: Content source (transcripts/abstracts/all)
            date_from: Filter from date
            date_to: Filter to date
            category: Category filter
            event_name: Event filter
            time_grouping: Time grouping (monthly/yearly/quarterly/none)
            top_n: Number of top results
            min_mentions: Minimum mention count

        Returns:
            Trend analysis results
        """
        return self._make_request("POST", "/api/v1/trends/analyze", {
            "analysis_type": analysis_type,
            "content_source": content_source,
            "date_from": date_from,
            "date_to": date_to,
            "category": category,
            "event_name": event_name,
            "time_grouping": time_grouping,
            "top_n": top_n,
            "min_mentions": min_mentions
        })

    def get_unique_values(
        self,
        event_name: bool = False,
        category_primary: bool = False,
        track: bool = False,
        company_name: bool = False,
        tech_level: bool = False,
        industries: bool = False
    ) -> Dict[str, Any]:
        """
        Get unique metadata values.

        Args:
            event_name: Get unique event names
            category_primary: Get unique categories
            track: Get unique tracks
            company_name: Get unique companies
            tech_level: Get unique tech levels
            industries: Get unique industries

        Returns:
            Unique values and counts
        """
        return self._make_request("POST", "/api/v1/trends/unique-values", {
            "event_name": event_name,
            "category_primary": category_primary,
            "track": track,
            "company_name": company_name,
            "tech_level": tech_level,
            "industries": industries
        })
