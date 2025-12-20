"""
Pydantic Request Models for MLOps Events API

These models define the structure of incoming API requests.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ===== Agent Query =====

class AgentQueryRequest(BaseModel):
    """Request model for agent natural language queries"""
    query: str = Field(
        ...,
        description="Natural language query",
        min_length=1,
        examples=["Show me the most popular talks from 2024"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation history. Auto-generated if not provided."
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response (for future use)"
    )


# ===== Semantic Search =====

class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(
        ...,
        description="Natural language search query",
        examples=["AI agents with memory", "MLOps deployment strategies"]
    )
    search_type: Literal["transcript", "meta", "bio", "all"] = Field(
        default="all",
        description="Content type to search"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter from this date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter to this date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category"
    )
    event_name: Optional[str] = Field(
        default=None,
        description="Filter by event name"
    )
    speaker_name: Optional[str] = Field(
        default=None,
        description="Filter by speaker name"
    )
    k_neighbors: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )


# ===== Metadata Filtering =====

class FilterTalksRequest(BaseModel):
    """Request model for filtering talks by metadata"""
    date_from: Optional[str] = Field(default=None, description="Filter from date")
    date_to: Optional[str] = Field(default=None, description="Filter to date")
    min_views: Optional[int] = Field(default=None, ge=0, description="Minimum view count")
    max_views: Optional[int] = Field(default=None, ge=0, description="Maximum view count")
    category: Optional[str] = Field(default=None, description="Category filter")
    event_name: Optional[str] = Field(default=None, description="Event name filter")
    track: Optional[str] = Field(default=None, description="Track filter")
    tech_level: Optional[int] = Field(default=None, ge=1, le=7, description="Technical level (1-7)")
    speaker_name: Optional[str] = Field(default=None, description="Speaker name filter")
    company_name: Optional[str] = Field(default=None, description="Company name filter")
    sort_by: Optional[Literal["date_desc", "date_asc", "views_desc", "views_asc", "title_asc", "tech_level_asc"]] = Field(
        default="date_desc",
        description="Sort order"
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return")


# ===== Speaker Analysis =====

class SpeakerAnalysisRequest(BaseModel):
    """Request model for speaker analytics"""
    speaker_name: Optional[str] = Field(
        default=None,
        description="Specific speaker to analyze (leave empty for all speakers)"
    )
    analysis_type: Literal["talk_count", "topics", "companies", "all"] = Field(
        default="all",
        description="Type of analysis to perform"
    )
    date_from: Optional[str] = Field(default=None, description="Filter from date")
    date_to: Optional[str] = Field(default=None, description="Filter to date")
    category: Optional[str] = Field(default=None, description="Category filter")
    min_talks: int = Field(
        default=1,
        ge=1,
        description="Minimum number of talks for inclusion in results"
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top speakers to return"
    )


# ===== Talk Details =====

class TalkDetailsRequest(BaseModel):
    """Request model for getting talk details"""
    talk_title: Optional[str] = Field(default=None, description="Talk title (exact match)")
    talk_id: Optional[str] = Field(default=None, description="Talk ID")
    include_transcript: bool = Field(default=False, description="Include full transcript")
    include_related: bool = Field(default=False, description="Include related talks")
    related_count: int = Field(default=5, ge=1, le=20, description="Number of related talks")
    time_start: Optional[int] = Field(default=None, ge=0, description="Transcript start time (seconds)")
    time_end: Optional[int] = Field(default=None, ge=0, description="Transcript end time (seconds)")


# ===== Similar Content =====

class SimilarContentRequest(BaseModel):
    """Request model for finding similar content"""
    reference_talk_title: Optional[str] = Field(default=None, description="Reference talk title")
    reference_talk_id: Optional[str] = Field(default=None, description="Reference talk ID")
    reference_query: Optional[str] = Field(default=None, description="Reference query text")
    similarity_type: Literal["content", "speaker", "topic", "all"] = Field(
        default="content",
        description="Type of similarity to search for"
    )
    exclude_same_speaker: bool = Field(default=False, description="Exclude talks by same speaker")
    date_from: Optional[str] = Field(default=None, description="Filter from date")
    date_to: Optional[str] = Field(default=None, description="Filter to date")
    category: Optional[str] = Field(default=None, description="Category filter")
    event_name: Optional[str] = Field(default=None, description="Event name filter")
    k_neighbors: int = Field(default=10, ge=1, le=50, description="Number of results")
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity score")


# ===== Trend Analysis =====

class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    analysis_type: Literal["tools", "topics", "technologies", "trends", "keywords"] = Field(
        ...,
        description="Type of analysis to perform"
    )
    content_source: Literal["transcripts", "abstracts", "all"] = Field(
        default="all",
        description="Content source for analysis"
    )
    date_from: Optional[str] = Field(default=None, description="Filter from date")
    date_to: Optional[str] = Field(default=None, description="Filter to date")
    category: Optional[str] = Field(default=None, description="Category filter")
    event_name: Optional[str] = Field(default=None, description="Event name filter")
    time_grouping: Optional[Literal["monthly", "yearly", "quarterly", "none"]] = Field(
        default="none",
        description="Group results by time period"
    )
    top_n: int = Field(default=20, ge=1, le=100, description="Number of top results")
    min_mentions: int = Field(default=2, ge=1, description="Minimum mention count")


# ===== Unique Values =====

class UniqueValuesRequest(BaseModel):
    """Request model for getting unique metadata values"""
    event_name: bool = Field(default=False, description="Get unique event names")
    category_primary: bool = Field(default=False, description="Get unique categories")
    track: bool = Field(default=False, description="Get unique tracks")
    company_name: bool = Field(default=False, description="Get unique companies")
    tech_level: bool = Field(default=False, description="Get unique tech levels")
    industries: bool = Field(default=False, description="Get unique industries")


# ===== Video Search =====

class VideoSearchRequest(BaseModel):
    """Request model for semantic video search"""
    query: str = Field(
        ...,
        description="Natural language query for video search",
        examples=["AI agents demonstration", "live coding session"]
    )
    top_n: int = Field(default=10, ge=1, le=50, description="Number of results")
    include_videos: bool = Field(
        default=False,
        description="Include video blobs in response (large payload)"
    )
