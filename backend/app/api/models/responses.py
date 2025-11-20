"""
Pydantic Response Models for MLOps Events API

These models define the structure of API responses.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


# ===== Base Response Models =====

class BaseResponse(BaseModel):
    """Base response model with success flag"""
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Optional message")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


# ===== Health Response =====

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Health status")
    version: str = Field(description="API version")


class DetailedHealthResponse(BaseModel):
    """Detailed health check with service status"""
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    services: Dict[str, str] = Field(description="Status of individual services")
    config: Dict[str, Any] = Field(description="Configuration information")


# ===== Agent Response =====

class AgentStep(BaseModel):
    """Agent execution step"""
    type: str = Field(description="Step type (tool_call, tool_result, ai_response)")
    content: Any = Field(description="Step content")
    tool_name: Optional[str] = Field(default=None, description="Tool name (for tool steps)")
    timestamp: Optional[str] = Field(default=None, description="Step timestamp")


class AgentQueryResponse(BaseModel):
    """Response from agent query"""
    success: bool = Field(description="Whether query was successful")
    answer: str = Field(description="Final answer from agent")
    steps: List[AgentStep] = Field(default=[], description="Execution steps")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


# ===== Talk Models =====

class TalkSummary(BaseModel):
    """Summary information about a talk (matches search_talks_by_filters output)"""
    talk_id: str
    title: str  # Changed from talk_title to match tool output
    speaker: str  # Changed from speaker_name to match tool output
    company: Optional[str] = None  # Changed from company_name
    category: Optional[str] = None  # Changed from category_primary
    event: Optional[str] = None  # Changed from event_name
    youtube_url: Optional[str] = None
    views: Optional[int] = None  # Changed from yt_views
    published_date: Optional[str] = None  # Changed from yt_published_at
    tech_level: Optional[int] = None
    track: Optional[str] = None
    abstract: Optional[str] = None


class SpeakerDetails(BaseModel):
    """Speaker details (nested in TalkDetail)"""
    name: Optional[str] = None
    bio: Optional[str] = None
    company: Optional[str] = None


class TalkDetail(BaseModel):
    """Detailed talk information (matches get_talk_details output)"""
    talk_id: str
    title: str
    speaker: str
    company: Optional[str] = None
    views: Optional[int] = None
    published_date: Optional[str] = None
    duration: Optional[str] = None
    youtube_url: Optional[str] = None
    event: Optional[str] = None
    category: Optional[str] = None
    track: Optional[str] = None
    tech_level: Optional[int] = None
    industries: Optional[str] = None
    keywords: Optional[str] = None
    abstract: Optional[str] = None
    speaker_details: Optional[SpeakerDetails] = None


class TranscriptChunk(BaseModel):
    """Transcript chunk (matches get_talk_details output)"""
    sequence: int = Field(description="Sequence number")
    text: str = Field(description="Chunk text")
    start_time: Optional[int] = Field(default=None, description="Start time in seconds")
    end_time: Optional[int] = Field(default=None, description="End time in seconds")
    duration: Optional[int] = Field(default=None, description="Duration in seconds")
    timestamp_formatted: Optional[str] = Field(default=None, description="Formatted timestamp (MM:SS)")


class SemanticSearchResult(BaseModel):
    """Semantic search result with similarity score"""
    talk_id: str
    title: Optional[str] = None  # Made optional - can be None if join fails
    speaker: Optional[str] = None  # Made optional - can be None if join fails
    youtube_url: Optional[str] = None
    event: Optional[str] = None
    category: Optional[str] = None
    similarity_score: float = Field(description="Similarity score (0-1)")
    matching_text: Optional[str] = Field(default=None, description="Matching text snippet")
    content_type: str = Field(description="Content type (transcript/meta/bio)")
    context_info: Optional[str] = None


# ===== Response Models =====

class SemanticSearchResponse(BaseModel):
    """Response from semantic search"""
    success: bool
    results: List[SemanticSearchResult]
    total_found: int
    search_summary: str
    query_vector_info: Optional[Any] = None  # Can be string or dict


class FilterTalksResponse(BaseModel):
    """Response from filtering talks (matches search_talks_by_filters output)"""
    success: bool
    results: List[TalkSummary]
    total_found: int
    query_summary: str
    sort_info: str  # Changed from filters_applied to match tool output


class RelatedTalk(BaseModel):
    """Related talk result (matches get_talk_details output for related talks)"""
    title: str
    speaker: str
    youtube_url: Optional[str] = None
    views: Optional[int] = None
    category: Optional[str] = None
    similarity_score: Optional[float] = None


class TalkDetailsResponse(BaseModel):
    """Response with detailed talk information (matches get_talk_details output)"""
    success: bool
    talk_info: TalkDetail  # Changed from "talk" to "talk_info"
    transcript_chunks: Optional[List[TranscriptChunk]] = None  # Changed from "transcript"
    related_talks: Optional[List[RelatedTalk]] = None
    summary: str  # Added to match tool output
    transcript_stats: Optional[Dict[str, Any]] = None


class SimilarTalkResult(BaseModel):
    """Similar talk result (matches find_similar_content output)"""
    talk_id: str
    title: str
    speaker: str
    youtube_url: Optional[str] = None
    category: Optional[str] = None
    views: Optional[int] = None
    similarity_score: float
    similarity_type: str
    similarity_reason: str
    matching_content: Optional[str] = None
    published_date: Optional[str] = None


class SimilarContentResponse(BaseModel):
    """Response from similar content search (matches find_similar_content output)"""
    success: bool
    similar_talks: List[SimilarTalkResult]  # Changed from "results" to "similar_talks"
    total_found: int
    reference_info: Dict[str, Any]
    similarity_analysis: str
    filters_applied: List[str]  # Changed from Dict to List to match tool output


# ===== Speaker Response =====

class SpeakerTalkItem(BaseModel):
    """Individual talk in speaker's talk list"""
    title: str
    date: Optional[str] = None
    views: Optional[int] = None
    event: Optional[str] = None
    category: Optional[str] = None
    youtube_url: Optional[str] = None
    abstract: Optional[str] = None


class SpeakerStats(BaseModel):
    """Speaker statistics (matches analyze_speaker_activity output)"""
    speaker_name: str
    total_talks: int
    total_views: int
    avg_views: int  # Changed from float to int to match tool output
    categories: List[str] = Field(alias="categories_covered")  # Accept both "categories" and "categories_covered"
    events: Optional[List[str]] = Field(default=None, alias="events_participated")  # Accept both "events" and "events_participated"
    companies: List[str]
    talk_list: Optional[List[SpeakerTalkItem]] = None  # Only present for specific speaker analysis

    class Config:
        populate_by_name = True  # Allow using either the field name or alias


class CompanyBreakdownItem(BaseModel):
    """Company breakdown item"""
    company: str
    speaker_count: int


class SpeakerAnalysisResponse(BaseModel):
    """Response from speaker analysis (matches analyze_speaker_activity output)"""
    success: bool
    speaker_stats: List[SpeakerStats]  # Changed from Optional single object to List
    company_breakdown: Optional[Union[Dict[str, Any], List[CompanyBreakdownItem]]] = None  # Can be dict (specific speaker) or list (general)
    topic_analysis: Optional[Dict[str, Any]] = None  # Added to match tool
    repeat_speakers: Optional[List[SpeakerStats]] = None  # Changed from List[str] to List[SpeakerStats]
    analysis_summary: str
    total_speakers: int  # Added to match tool output


# ===== Trend Response =====

class TrendItem(BaseModel):
    """Trend analysis item (matches analyze_topics_and_trends output)"""
    item: str = Field(description="Item name (tool, topic, technology, etc.)")
    count: int = Field(description="Mention count")
    percentage: float = Field(description="Percentage of total")
    sample_mentions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sample mentions (for tools)")


class TrendAnalysisResponse(BaseModel):
    """Response from trend analysis (matches analyze_topics_and_trends output)"""
    success: bool
    analysis_results: List[TrendItem]  # Changed from "results" to "analysis_results"
    total_items_found: int
    analysis_summary: str  # Changed from analysis_type to analysis_summary
    content_stats: Dict[str, Any]
    time_trends: Optional[Dict[str, Any]] = None  # Changed from time_grouping to time_trends


# ===== Unique Values Response =====

class UniqueValuesResponse(BaseModel):
    """Response with unique metadata values (matches get_unique_values output)"""
    success: bool
    unique_values: Dict[str, List[Any]]  # Changed from "values" to "unique_values"
    counts: Dict[str, int]
    total_entities: Optional[int] = None  # Added to match tool output
    properties_queried: Optional[List[str]] = None  # Added to match tool output


# ===== Video Response =====

class VideoMetadata(BaseModel):
    """Video metadata"""
    fps: Optional[float] = None
    duration_sec: Optional[int] = None
    resolution: Optional[str] = None


class VideoSearchResult(BaseModel):
    """Video search result"""
    talk_title: str
    speaker_name: str
    company_name: Optional[str] = None
    category_primary: Optional[str] = None
    youtube_url: Optional[str] = None
    youtube_id: Optional[str] = None
    yt_views: Optional[int] = None
    distance: Optional[float] = None
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None
    video_blob: Optional[bytes] = Field(default=None, description="Video binary data (if requested)")


class VideoSearchResponse(BaseModel):
    """Response from video search"""
    success: bool
    results: List[VideoSearchResult]
    total_found: int
    search_summary: str  # Changed from query_summary to match tool output


# ===== List Response (Generic) =====

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    success: bool
    results: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int
