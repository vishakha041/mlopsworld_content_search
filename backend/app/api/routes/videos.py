"""
Video Endpoints

Endpoints for video search and retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
from app.api.models.requests import VideoSearchRequest
from app.api.models.responses import VideoSearchResponse
from app.dependencies import get_db_connector, get_twelvelabs_client, verify_api_key_skip_options
from app.tools.utils import create_db_connector

router = APIRouter()


@router.get("/stream")
def stream_video(
    talk_title: str,
    background_tasks: BackgroundTasks
):
    """
    Stream video content for a given talk title using FileResponse.
    This handles Range headers automatically for robust video playback.
    """
    print(f"Streaming video for talk: {talk_title}")
    
    # Create a fresh DB connector for this request to ensure thread safety
    # and avoid "Bad file descriptor" errors with the shared connection
    db = create_db_connector()
    
    try:
        # Query for the video
        # Explicitly request MP4 container and H.264 codec for browser compatibility
        query = [
            {
                "FindVideo": {
                    "constraints": {"talk_title": ["==", talk_title]},
                    "blobs": True,
                    "results": {"limit": 1}
                }
            }
        ]
        
        results, blobs = db.query(query)
        
        if not blobs or len(blobs) == 0:
            print(f"Video not found for talk: {talk_title}")
            raise HTTPException(status_code=404, detail="Video not found")
            
        blob = blobs[0]
        
        if blob is None:
            print(f"Video blob is None for talk: {talk_title}")
            raise HTTPException(status_code=404, detail="Video content unavailable")
            
        video_bytes = blob
        video_size = len(video_bytes)
        print(f"Found video blob of size: {video_size} bytes")
        
        # Create a temp file to serve via FileResponse
        # This mimics the robust behavior of the Streamlit app
        fd, path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(video_bytes)
            
        # Schedule cleanup to remove the temp file after response is sent
        background_tasks.add_task(os.remove, path)
        
        # FileResponse automatically handles Range headers and streaming
        return FileResponse(
            path, 
            media_type="video/mp4", 
            filename="video.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error streaming video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")


@router.post("/search", response_model=VideoSearchResponse, dependencies=[Depends(verify_api_key_skip_options)])
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
