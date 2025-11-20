"""
Shared Test Configuration

Provides common fixtures and configuration for both unit and integration tests.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest


def pytest_configure(config):
    """
    Register custom markers for test organization.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests with mocked dependencies (fast)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with real resources (slow)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (agent queries, complex operations)"
    )


@pytest.fixture
def api_key():
    """
    Provides API key for tests.

    Returns the API secret key from settings.
    """
    from app.config import settings
    return settings.API_SECRET_KEY


@pytest.fixture
def auth_headers(api_key: str):
    """
    Provides authentication headers.

    Returns headers dict with API key for authenticated requests.
    """
    return {"X-API-Key": api_key}
