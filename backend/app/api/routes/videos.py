"""
Video Endpoints

Endpoints for video search and retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import VideoSearchRequest
from app.api.models.responses import VideoSearchResponse
from app.dependencies import get_db_connector, get_twelvelabs_client, verify_api_key

router = APIRouter()


@router.post("/search", response_model=VideoSearchResponse, dependencies=[Depends(verify_api_key)])
async def search_videos_semantically(
    request: VideoSearchRequest,
    db = Depends(get_db_connector),
    tl_client = Depends(get_twelvelabs_client)
):
    """
    Semantic video search using Twelve Labs embeddings.

    Searches video content using visual and audio understanding,
    going beyond what's said in the transcript to understand
    presentation style, demos, and visual content.

    **Authentication Required**: Include `X-API-Key` header

    **Example Queries**:
    - "AI agents demonstration"
    - "live coding session"
    - "presentations with architecture diagrams"
    - "talks with product demos"

    Args:
        request: VideoSearchRequest with query and options

    Returns:
        VideoSearchResponse with matching videos and similarity scores

    Note:
        Requires Twelve Labs API key to be configured.
    """
    try:
        from app.tools.search_videos_semantically import search_videos_semantically as video_search_tool

        result = video_search_tool.invoke(request.dict())

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Video search failed"))

        return VideoSearchResponse(**result)

    except HTTPException as e:
        # Re-raise HTTP exceptions (like 503 from missing TL client)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video search failed: {str(e)}")
