"""
Health Check Endpoints

Endpoints for checking API and service health.
"""

from fastapi import APIRouter, Request, BackgroundTasks
from app.api.models.responses import HealthResponse, DetailedHealthResponse
from app.config import settings
import asyncio

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        HealthResponse with status and version
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@router.get("/api/v1/status", response_model=DetailedHealthResponse)
async def detailed_status(request: Request):
    """
    Detailed status check with service information.

    Returns:
        DetailedHealthResponse with detailed service status
    """
    services = {}

    # Check database connector
    if hasattr(request.app.state, "db_connector") and request.app.state.db_connector:
        services["database"] = "connected"
    else:
        services["database"] = "not_initialized"

    # Check embedding model
    if hasattr(request.app.state, "embedding_model") and request.app.state.embedding_model:
        services["embedding_model"] = "loaded"
    else:
        services["embedding_model"] = "not_loaded"

    # Check Twelve Labs client
    if hasattr(request.app.state, "twelvelabs_client"):
        if request.app.state.twelvelabs_client:
            services["twelvelabs_client"] = "connected"
        else:
            services["twelvelabs_client"] = "not_configured"
    else:
        services["twelvelabs_client"] = "not_initialized"

    # Check agent
    if hasattr(request.app.state, "agent") and request.app.state.agent:
        services["agent"] = "ready"
    else:
        services["agent"] = "not_initialized"

    # Overall status
    all_critical_services_ready = all(
        services.get(s) in ["connected", "loaded", "ready"]
        for s in ["database", "embedding_model", "agent"]
    )

    status = "healthy" if all_critical_services_ready else "degraded"

    return DetailedHealthResponse(
        status=status,
        version="1.0.0",
        services=services,
        config={
            "model_name": settings.MODEL_NAME,
            "embed_model": settings.EMBED_MODEL,
            "max_iterations": settings.MAX_ITERATIONS
        }
    )


@router.post("/api/v1/warmup")
async def warmup(request: Request, background_tasks: BackgroundTasks):
    """
    Warmup endpoint to initialize all resources.
    
    Call this after deployment to pre-load models and connections
    so that actual user requests don't timeout.
    
    Returns immediately but initializes resources in background.
    """
    def initialize_all():
        """Initialize all resources synchronously"""
        try:
            from app.dependencies import get_db_connector, get_embedding_model, get_agent
            
            # These will trigger lazy initialization
            print("Warmup: Loading database connector...")
            get_db_connector(request)
            
            print("Warmup: Loading embedding model...")
            get_embedding_model(request)
            
            print("Warmup: Loading agent...")
            get_agent(request)
            
            print("Warmup complete - all resources loaded!")
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    # Run initialization in background
    background_tasks.add_task(initialize_all)
    
    return {
        "status": "warmup_started",
        "message": "Resources are being initialized in the background. Check /api/v1/status to see progress."
    }
