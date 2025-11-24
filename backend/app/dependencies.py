"""
Dependency Injection for FastAPI

This module provides dependency injection functions for shared resources:
- Database connector
- Embedding model
- Twelve Labs client
- LangGraph agent
"""

from fastapi import FastAPI, Request, Depends, HTTPException, Header
from typing import Optional
import threading

from app.config import settings

# Thread lock for lazy initialization
_init_lock = threading.Lock()


async def initialize_resources(app: FastAPI):
    """
    Initialize all shared resources at application startup.

    Resources are stored in app.state and accessed via dependency injection.
    """
    # Check if already initialized (for tests with pre-loaded fixtures)
    if (hasattr(app.state, 'db_connector') and app.state.db_connector is not None and
        hasattr(app.state, 'embedding_model') and app.state.embedding_model is not None and
        hasattr(app.state, 'agent') and app.state.agent is not None):
        print("   Resources already initialized")
        return

    # Import here to avoid circular imports and ensure modules are available
    from app.tools.utils import (
        create_db_connector,
        create_embedding_model,
        create_twelvelabs_client
    )
    from app.agent.agent import create_mlops_agent

    # Initialize database connector
    print("   Initializing database connector...")
    app.state.db_connector = create_db_connector()

    # Initialize embedding model
    print("   Loading embedding model...")
    app.state.embedding_model = create_embedding_model()

    # Initialize Twelve Labs client (optional)
    if settings.TL_API_KEY:
        print("   Initializing Twelve Labs client...")
        app.state.twelvelabs_client = create_twelvelabs_client()
    else:
        print("   Twelve Labs API key not found - video search will be unavailable")
        app.state.twelvelabs_client = None

    # Initialize agent
    print("   Creating LangGraph agent...")
    app.state.agent = create_mlops_agent()

    print("   All resources initialized")


async def cleanup_resources(app: FastAPI):
    """
    Cleanup resources at application shutdown.
    """
    # Database cleanup
    if hasattr(app.state, "db_connector") and app.state.db_connector:
        print("   Closing database connection...")
        # Add any necessary cleanup for ApertureDB connector if needed
        app.state.db_connector = None

    # Model cleanup
    if hasattr(app.state, "embedding_model"):
        app.state.embedding_model = None

    # Client cleanup
    if hasattr(app.state, "twelvelabs_client"):
        app.state.twelvelabs_client = None

    # Agent cleanup
    if hasattr(app.state, "agent"):
        app.state.agent = None


# ===== Dependency Injection Functions =====

def get_db_connector(request: Request):
    """Get database connector from application state (lazy init)"""
    if not hasattr(request.app.state, "db_connector") or request.app.state.db_connector is None:
        with _init_lock:
            # Double-check after acquiring lock
            if not hasattr(request.app.state, "db_connector") or request.app.state.db_connector is None:
                from app.tools.utils import create_db_connector
                print("   Lazy-loading database connector...")
                request.app.state.db_connector = create_db_connector()
    return request.app.state.db_connector


def get_embedding_model(request: Request):
    """Get embedding model from application state (lazy init)"""
    if not hasattr(request.app.state, "embedding_model") or request.app.state.embedding_model is None:
        with _init_lock:
            # Double-check after acquiring lock
            if not hasattr(request.app.state, "embedding_model") or request.app.state.embedding_model is None:
                from app.tools.utils import create_embedding_model
                print("   Lazy-loading embedding model...")
                request.app.state.embedding_model = create_embedding_model()
    return request.app.state.embedding_model


def get_twelvelabs_client(request: Request):
    """Get Twelve Labs client from application state (lazy init)"""
    if not hasattr(request.app.state, "twelvelabs_client"):
        with _init_lock:
            # Double-check after acquiring lock
            if not hasattr(request.app.state, "twelvelabs_client"):
                if settings.TL_API_KEY:
                    from app.tools.utils import create_twelvelabs_client
                    print("   Lazy-loading Twelve Labs client...")
                    request.app.state.twelvelabs_client = create_twelvelabs_client()
                else:
                    request.app.state.twelvelabs_client = None
    
    if request.app.state.twelvelabs_client is None:
        raise HTTPException(
            status_code=503,
            detail="Twelve Labs client not available - API key not configured"
        )
    return request.app.state.twelvelabs_client


def get_agent(request: Request):
    """Get LangGraph agent from application state (lazy init)"""
    if not hasattr(request.app.state, "agent") or request.app.state.agent is None:
        with _init_lock:
            # Double-check after acquiring lock
            if not hasattr(request.app.state, "agent") or request.app.state.agent is None:
                from app.agent.agent import create_mlops_agent
                print("   Lazy-loading agent...")
                request.app.state.agent = create_mlops_agent()
    return request.app.state.agent


# ===== Authentication =====

async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verify API key from request headers.

    In production, you should use a more secure method like JWT tokens.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header in your request."
        )

    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return x_api_key
