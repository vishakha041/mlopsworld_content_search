"""
Talk Endpoints

Endpoints for searching, filtering, and retrieving talk information.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import (
    SemanticSearchRequest,
    FilterTalksRequest,
    TalkDetailsRequest,
    SimilarContentRequest
)
from app.api.models.responses import (
    SemanticSearchResponse,
    FilterTalksResponse,
    TalkDetailsResponse,
    SimilarContentResponse
)
from app.dependencies import get_db_connector, get_embedding_model, verify_api_key

router = APIRouter()


@router.post("/search", response_model=SemanticSearchResponse, dependencies=[Depends(verify_api_key)])
async def search_talks_semantically(
    request: SemanticSearchRequest,
    db = Depends(get_db_connector),
    model = Depends(get_embedding_model)
):
    """
    Semantic search across talks using natural language queries.

    Searches through video transcripts, talk abstracts, and speaker bios
    using embeddings to find semantically similar content.

    **Authentication Required**: Include `X-API-Key` header

    **Example Queries**:
    - "AI agents with memory"
    - "MLOps deployment strategies"
    - "vector databases in production"

    Args:
        request: SemanticSearchRequest with query and filters

    Returns:
        SemanticSearchResponse with matching talks and similarity scores
    """
    try:
        from app.tools.search_talks_semantically import search_talks_semantically as search_tool

        # Filter out None values and Swagger placeholder values
        params = {k: v for k, v in request.dict().items()
                  if v is not None and v != "string" and v != ""}

        result = search_tool.invoke(params)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))

        return SemanticSearchResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.post("/filter", response_model=FilterTalksResponse, dependencies=[Depends(verify_api_key)])
async def filter_talks(
    request: FilterTalksRequest,
    db = Depends(get_db_connector)
):
    """
    Filter talks by metadata (dates, views, categories, speakers, companies).

    **Authentication Required**: Include `X-API-Key` header

    **Example Filters**:
    - Date range: 2024-01-01 to 2024-12-31
    - Minimum views: 1000
    - Category: MLOps
    - Speaker: John Doe
    - Sort by: views_desc

    Args:
        request: FilterTalksRequest with filter criteria

    Returns:
        FilterTalksResponse with matching talks
    """
    try:
        from app.tools.search_talks_by_filters import search_talks_by_filters as filter_tool

        # Transform sort_by to separate fields (e.g., "date_desc" -> sort_by="date", sort_order="desc")
        params = request.dict()
        sort_by = params.get("sort_by", "date_desc")

        if "_" in sort_by:
            field, order = sort_by.rsplit("_", 1)
            params["sort_by"] = field
            params["sort_order"] = order
        else:
            params["sort_by"] = sort_by
            params["sort_order"] = "desc"

        result = filter_tool.invoke(params)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Filtering failed"))

        return FilterTalksResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Talk filtering failed: {str(e)}")


@router.post("/details", response_model=TalkDetailsResponse, dependencies=[Depends(verify_api_key)])
async def get_talk_details(
    request: TalkDetailsRequest,
    db = Depends(get_db_connector)
):
    """
    Get detailed information about a specific talk.

    **Authentication Required**: Include `X-API-Key` header

    Optionally includes:
    - Full transcript with timestamps
    - Related talks based on semantic similarity
    - Transcript statistics

    Args:
        request: TalkDetailsRequest with talk title or ID

    Returns:
        TalkDetailsResponse with detailed talk information
    """
    try:
        from app.tools.get_talk_details import get_talk_details as details_tool

        if not request.talk_title and not request.talk_id:
            raise HTTPException(status_code=400, detail="Either talk_title or talk_id is required")

        # Filter out None values to avoid tool validation errors
        params = {k: v for k, v in request.dict().items() if v is not None}

        result = details_tool.invoke(params)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get talk details"))

        return TalkDetailsResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get talk details: {str(e)}")


@router.post("/similar", response_model=SimilarContentResponse, dependencies=[Depends(verify_api_key)])
async def find_similar_talks(
    request: SimilarContentRequest,
    db = Depends(get_db_connector),
    model = Depends(get_embedding_model)
):
    """
    Find talks similar to a reference talk or query.

    **Authentication Required**: Include `X-API-Key` header

    **Search Options**:
    - By reference talk (title or ID)
    - By reference query text
    - Filter by similarity type (content, speaker, topic)
    - Exclude same speaker
    - Apply date/category filters

    Args:
        request: SimilarContentRequest with reference and filters

    Returns:
        SimilarContentResponse with similar talks
    """
    try:
        from app.tools.find_similar_content import find_similar_content as similar_tool

        if not any([request.reference_talk_title, request.reference_talk_id, request.reference_query]):
            raise HTTPException(
                status_code=400,
                detail="One of reference_talk_title, reference_talk_id, or reference_query is required"
            )

        # Filter out None values to avoid tool validation errors
        params = {k: v for k, v in request.dict().items() if v is not None}

        result = similar_tool.invoke(params)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to find similar content"))

        return SimilarContentResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar content: {str(e)}")
