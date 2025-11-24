"""
Authentication Middleware

API key authentication for the FastAPI application.
"""

from fastapi import Header, HTTPException, status, Request
from typing import Optional
from app.config import settings


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verify API key from request headers.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header in your request.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return x_api_key


async def verify_api_key_skip_options(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Verify API key but skip authentication for OPTIONS (preflight) requests.
    This allows CORS preflight requests to succeed without authentication.
    
    Args:
        request: The incoming request
        x_api_key: API key from X-API-Key header
    
    Returns:
        The validated API key or None for OPTIONS requests
    
    Raises:
        HTTPException: If API key is missing or invalid (except for OPTIONS)
    """
    # Skip auth for OPTIONS (preflight) requests
    if request.method == "OPTIONS":
        return None
    
    # Require auth for all other requests
    return await verify_api_key(x_api_key)


class OptionalAuth:
    """
    Optional authentication - allows requests with or without API key.
    Useful for public endpoints that have enhanced features with auth.
    """

    async def __call__(self, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        if x_api_key and x_api_key != settings.API_SECRET_KEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        return x_api_key
