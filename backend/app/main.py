"""
MLOps Events FastAPI Backend

Main FastAPI application with lifespan resource management.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.middleware.errors import setup_exception_handlers

print("Starting MLOps Events API...")
print("Resources will be loaded lazily on first use")
print("API binding to port...")

# Create FastAPI app (no blocking lifespan)
app = FastAPI(
    title="MLOps Events API",
    description="AI-powered semantic search and analysis platform for MLOps conference talks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# No auto-warmup - resources load lazily on first request only
# This prevents deployment timeouts on free tier hosting

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add middleware to handle OPTIONS requests
@app.middleware("http")
async def handle_options(request: Request, call_next):
    """Handle OPTIONS preflight requests"""
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, X-API-Key, Authorization",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    return await call_next(request)

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
