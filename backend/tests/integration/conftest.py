"""
Integration Test Fixtures

Provides real resources for integration testing:
- Real ApertureDB connection
- Real embedding model
- Real LangGraph agent
- Real test client
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_db_connector, get_embedding_model, get_agent
from app.config import settings


@pytest.fixture(scope="session")
def real_db_connector():
    """
    Provides a real ApertureDB connector.

    This connects to the actual database configured in .env
    Scope is 'session' to reuse connection across all tests.
    """
    from app.tools.utils import create_db_connector
    connector = create_db_connector()
    yield connector
    # Cleanup if needed


@pytest.fixture(scope="session")
def real_embedding_model():
    """
    Provides a real embedding model.

    Loads the actual SentenceTransformer model.
    Scope is 'session' because loading the model is expensive.
    """
    from app.tools.utils import create_embedding_model
    model = create_embedding_model()
    yield model
    # Cleanup if needed


@pytest.fixture(scope="session")
def real_agent():
    """
    Provides a real LangGraph agent.

    Creates the actual MLOps agent with all tools.
    Scope is 'session' to reuse agent across tests.
    """
    from app.agent.agent import create_mlops_agent
    agent = create_mlops_agent()
    yield agent
    # Cleanup if needed


@pytest.fixture
def integration_client(real_db_connector, real_embedding_model, real_agent):
    """
    Provides a test client with real dependencies injected.

    This overrides the dependency injection to use real resources
    instead of the ones created during app startup.
    """
    # Pre-populate app.state to prevent lifespan from re-initializing
    app.state.db_connector = real_db_connector
    app.state.embedding_model = real_embedding_model
    app.state.agent = real_agent
    app.state.twelvelabs_client = None  # Not needed for most tests

    # Override dependencies to use our real fixtures
    app.dependency_overrides[get_db_connector] = lambda: real_db_connector
    app.dependency_overrides[get_embedding_model] = lambda: real_embedding_model
    app.dependency_overrides[get_agent] = lambda: real_agent

    with TestClient(app) as client:
        yield client

    # Clear overrides and state after test
    app.dependency_overrides.clear()
    app.state.db_connector = None
    app.state.embedding_model = None
    app.state.agent = None
    app.state.twelvelabs_client = None


@pytest.fixture
def api_key() -> str:
    """
    Provides the API key for authenticated requests.

    Returns the actual API key from settings.
    """
    return settings.API_SECRET_KEY


@pytest.fixture
def auth_headers(api_key: str) -> dict:
    """
    Provides authentication headers for requests.

    Returns headers with the API key.
    """
    return {"X-API-Key": api_key}
