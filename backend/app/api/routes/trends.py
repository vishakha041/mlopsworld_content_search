"""
Trend Analysis Endpoints

Endpoints for analyzing topics, tools, technologies, and trends.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.api.models.requests import TrendAnalysisRequest, UniqueValuesRequest
from app.api.models.responses import TrendAnalysisResponse, UniqueValuesResponse
from app.dependencies import get_db_connector, verify_api_key

router = APIRouter()


@router.post("/analyze", response_model=TrendAnalysisResponse, dependencies=[Depends(verify_api_key)])
async def analyze_trends(
    request: TrendAnalysisRequest,
    db = Depends(get_db_connector)
):
    """
    Analyze topics, tools, technologies, and trends across talks.

    **Authentication Required**: Include `X-API-Key` header

    **Analysis Types**:
    - `tools`: Software and libraries (e.g., LangChain, TensorFlow, Docker)
    - `topics`: Domain topics (e.g., deployment, monitoring, training)
    - `technologies`: Technology concepts (e.g., ML, LLM, RAG, CI/CD)
    - `trends`: General trends over time
    - `keywords`: Frequent keywords and terms

    **Content Sources**:
    - `transcripts`: Analyze video transcripts
    - `abstracts`: Analyze talk abstracts and metadata
    - `all`: Analyze all content

    **Time Grouping**:
    - `monthly`: Group by month
    - `yearly`: Group by year
    - `quarterly`: Group by quarter
    - `none`: No time grouping

    **Use Cases**:
    - "What tools are trending in MLOps?"
    - "Popular topics in 2024"
    - "Technology adoption over time"
    - "Most mentioned keywords"

    Args:
        request: TrendAnalysisRequest with analysis type and filters

    Returns:
        TrendAnalysisResponse with trend analysis results
    """
    try:
        from app.tools.analyze_topics_and_trends import analyze_topics_and_trends as trends_tool

        result = trends_tool.invoke(request.dict())

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Trend analysis failed"))

        # No transformation needed - response model now matches tool output
        return TrendAnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


@router.post("/unique-values", response_model=UniqueValuesResponse, dependencies=[Depends(verify_api_key)])
async def get_unique_values(
    request: UniqueValuesRequest,
    db = Depends(get_db_connector)
):
    """
    Get unique values for talk metadata properties.

    **Authentication Required**: Include `X-API-Key` header

    **Available Properties**:
    - `event_name`: All event names in the database
    - `category_primary`: All primary categories
    - `track`: All conference tracks
    - `company_name`: All companies represented
    - `tech_level`: All technical levels (1-7)
    - `industries`: All industries

    **Use Cases**:
    - "What events are in the database?"
    - "Show me all categories"
    - "Which companies have presented?"
    - "What technical levels are available?"

    Args:
        request: UniqueValuesRequest with boolean flags for each property

    Returns:
        UniqueValuesResponse with unique values and counts
    """
    try:
        from app.tools.get_unique_values import get_unique_values as values_tool

        result = values_tool.invoke(request.dict())

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get unique values"))

        return UniqueValuesResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get unique values: {str(e)}")
