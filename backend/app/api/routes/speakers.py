"""
Speaker Endpoints

Endpoints for speaker analytics and activity analysis.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import SpeakerAnalysisRequest
from app.api.models.responses import SpeakerAnalysisResponse
from app.dependencies import get_db_connector, verify_api_key_skip_options

router = APIRouter()


@router.post("/analyze", response_model=SpeakerAnalysisResponse, dependencies=[Depends(verify_api_key_skip_options)])
async def analyze_speaker_activity(
    request: SpeakerAnalysisRequest,
    db = Depends(get_db_connector)
):
    """
    Analyze speaker activity, frequency, and company representation.

    **Authentication Required**: Include `X-API-Key` header

    **Analysis Types**:
    - `talk_count`: Speaker talk counts and ranking
    - `topics`: Topics covered by speakers
    - `companies`: Company breakdown and representation
    - `all`: All of the above

    **Use Cases**:
    - "Who are the most active speakers?"
    - "Which companies have presented the most?"
    - "Find repeat speakers with 2+ talks"
    - "Analyze a specific speaker's activity"

    Args:
        request: SpeakerAnalysisRequest with speaker name and analysis type

    Returns:
        SpeakerAnalysisResponse with speaker statistics and analytics
    """
    try:
        from app.tools.analyze_speaker_activity import analyze_speaker_activity as analysis_tool

        result = analysis_tool.invoke(request.dict())

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        return SpeakerAnalysisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speaker analysis failed: {str(e)}")
