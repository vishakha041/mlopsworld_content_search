"""
CORS Middleware Configuration

Configuration for Cross-Origin Resource Sharing (CORS).
"""

from fastapi.middleware.cors import CORSMiddleware
from app.config import settings


def get_cors_config():
    """
    Get CORS configuration for the FastAPI application.

    Returns:
        Dictionary with CORS configuration
    """
    return {
        "allow_origins": settings.get_allowed_origins_list(),
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "Accept",
            "Origin",
            "User-Agent"
        ],
        "expose_headers": ["Content-Length", "X-Request-ID"],
        "max_age": 3600  # Cache preflight requests for 1 hour
    }
