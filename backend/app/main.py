"""
MLOps Events FastAPI Backend

Main FastAPI application with lifespan resource management.
"""

from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.middleware.errors import setup_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for resource initialization and cleanup.

    Resources initialized once at startup and shared across all requests:
    - Database connector
    - Embedding model
    - Twelve Labs client
    - LangGraph agent
    """
    print("🚀 Starting MLOps Events API...")
    print("📦 Loading resources...")

    # Import here to avoid circular imports
    from app.dependencies import initialize_resources, cleanup_resources

    # Kick off resource initialization in the background so startup doesn't block
    asyncio.create_task(initialize_resources(app))

    print("✅ Resources loaded successfully!")
    print(f"🌐 API starting; binding to provided PORT")
    print(f"📚 Docs at /docs once live")

    yield

    # Shutdown - cleanup resources
    print("🛑 Shutting down...")
    await cleanup_resources(app)
    print("✅ Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="MLOps Events API",
    description="AI-powered semantic search and analysis platform for MLOps conference talks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Import and include routers
from app.api.routes import health, agent, talks, speakers, videos, trends

app.include_router(health.router, tags=["Health"])
app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])
app.include_router(talks.router, prefix="/api/v1/talks", tags=["Talks"])
app.include_router(speakers.router, prefix="/api/v1/speakers", tags=["Speakers"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["Videos"])
app.include_router(trends.router, prefix="/api/v1/trends", tags=["Trends"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MLOps Events API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
