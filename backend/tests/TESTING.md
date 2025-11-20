# Testing Guide

Complete guide for testing the MLOps Events API with integration tests.

## Quick Start

```bash
# Install test dependencies
cd backend
pip install -r requirements-test.txt

# Run all tests
pytest

# Run integration tests
pytest tests/integration
```

## Test Structure

```
tests/
├── integration/                   # Integration tests (real resources)
│   ├── conftest.py               # Integration test fixtures
│   ├── test_health_integration.py
│   ├── test_talks_integration.py
│   ├── test_semantic_search_integration.py
│   ├── test_speakers_integration.py
│   ├── test_trends_integration.py
│   └── test_agent_integration.py
└── conftest.py                    # Shared configuration
```

## Integration Tests

**Purpose:** Test complete system with real resources

**Characteristics:**
- Use real ApertureDB connection
- Use real embedding model
- Use real LangGraph agent
- Slower execution (~2-5 minutes)
- Run before deployment
- Test actual functionality

**What They Test:**
- Real database queries
- Actual semantic search with embeddings
- Agent tool selection and execution
- End-to-end workflows
- Data accuracy

**Requirements:**
- Valid API keys in .env (GOOGLE_API_KEY, APERTUREDB_KEY, etc.)
- Access to ApertureDB instance
- Real data in database (280 talks)

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run integration tests
pytest tests/integration

# Run with verbose output
pytest -v

# Run specific file
pytest tests/integration/test_semantic_search_integration.py

# Run specific test
pytest tests/integration/test_semantic_search_integration.py::test_semantic_search_all_types

# Stop at first failure
pytest -x
```

### Using Markers

```bash
# Run all integration tests
pytest -m integration

# Run all slow tests
pytest -m slow

# Exclude slow tests
pytest -m "not slow"

# Run integration but skip slow tests
pytest -m "integration and not slow"
```

## Test Anatomy

### Integration Test Example

```python
def test_semantic_search_all_types(integration_client: TestClient, auth_headers: dict):
    """
    Test semantic search with real embeddings and database.

    Uses real embedding model to generate query vector.
    Makes actual database similarity search.
    """
    # No mocking - uses real resources
    response = integration_client.post(
        "/api/v1/talks/search",
        headers=auth_headers,
        json={
            "query": "machine learning deployment",
            "search_type": "all",
            "k_neighbors": 10
        }
    )

    # Verify real results
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    if data["total_found"] > 0:
        result = data["results"][0]
        assert 0.0 <= result["similarity_score"] <= 1.0
```

## Fixtures

### Shared Fixtures (tests/conftest.py)

Available in all tests:

```python
# API key from settings
def my_test(api_key: str):
    pass

# Authentication headers
def my_test(auth_headers: dict):
    pass
```

### Integration Test Fixtures (tests/integration/conftest.py)

Available only in integration tests:

```python
# Test client with real resources
def my_test(integration_client: TestClient):
    pass

# Real database connector
def my_test(real_db_connector):
    pass

# Real embedding model
def my_test(real_embedding_model):
    pass

# Real agent
def my_test(real_agent):
    pass
```

## Adding New Tests

### Adding an Integration Test

1. Create file in `tests/integration/` ending with `_integration.py`
2. Use `integration_client` fixture
3. No mocking - test real functionality
4. Mark with `pytestmark = pytest.mark.integration`
5. Mark slow tests with `@pytest.mark.slow`

Example:

```python
pytestmark = pytest.mark.integration

@pytest.mark.slow
def test_real_functionality(integration_client: TestClient, auth_headers: dict):
    """Test with real resources."""
    # No mocking
    response = integration_client.post("/endpoint", headers=auth_headers, json={...})

    assert response.status_code == 200
    # Verify real data
```

## Common Patterns

### Testing Success Response

```python
def test_success(integration_client: TestClient, auth_headers: dict):
    response = integration_client.post("/endpoint", headers=auth_headers, json={...})
    assert response.status_code == 200
    assert response.json()["success"] is True
```

### Testing Validation Errors

```python
def test_missing_field(integration_client: TestClient, auth_headers: dict):
    response = integration_client.post("/endpoint", headers=auth_headers, json={})
    assert response.status_code == 422
```

### Testing Authentication

```python
def test_no_auth(integration_client: TestClient):
    response = integration_client.post("/endpoint", json={...})
    assert response.status_code == 401
```

### Parametrized Tests

```python
@pytest.mark.parametrize("value,expected", [
    ("valid", 200),
    ("invalid", 422),
])
def test_values(integration_client, auth_headers, value, expected):
    response = integration_client.post("/endpoint", headers=auth_headers, json={"field": value})
    assert response.status_code == expected
```

## Troubleshooting

### Import Errors

```bash
# Ensure you're in backend directory
cd backend

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### "No module named 'app'"

Run pytest from `backend/` directory, not `backend/tests/`:

```bash
cd backend
pytest
```

### Integration Tests Fail

Check that:

1. API keys are set in `backend/.env`:
   ```
   GOOGLE_API_KEY=...
   APERTUREDB_KEY=...
   TL_API_KEY=...
   ```

2. Database connection works:
   ```bash
   cd backend
   python -c "from app.tools.utils import create_db_connector; print(create_db_connector())"
   ```

3. Embedding model loads:
   ```bash
   python -c "from app.tools.utils import create_embedding_model; print(create_embedding_model())"
   ```

### Slow Integration Tests

Integration tests are inherently slower because they:
- Connect to real database
- Load embedding models
- Execute agent reasoning
- Process real queries

Expected times:
- Simple queries: 1-5 seconds
- Agent queries: 10-60 seconds
- Full integration suite: 2-5 minutes

To skip slow tests:
```bash
pytest tests/integration -m "not slow"
```

## Test Organization

### Test Files by Category

**Integration Tests:**
- `test_health_integration.py` - Health with real services
- `test_talks_integration.py` - Talk queries with real DB
- `test_semantic_search_integration.py` - Semantic search with real embeddings
- `test_speakers_integration.py` - Speaker analysis with real DB
- `test_trends_integration.py` - Trend analysis with real DB
- `test_agent_integration.py` - Agent with real tools and DB

### Test Coverage Goals

| Component | Integration Coverage |
|-----------|---------------------|
| Tools | End-to-end |
| Agent | End-to-end |
| API Routes | End-to-end |

## Best Practices

### DO:
- Write clear test names describing what's tested
- Add docstrings explaining verification points
- Test both success and error cases
- Use parametrize for similar test cases
- Keep tests independent (no shared state)
- Run integration tests before deployment

### DON'T:
- Test implementation details
- Create tests that depend on execution order
- Ignore failing tests
- Write tests without assertions
- Use mocks in integration tests
- Commit without running tests

## Continuous Integration

### Recommended CI Pipeline

```yaml
# Run before merge/deploy
- pytest tests/integration -m "not slow"

# Run nightly or on release
- pytest tests/integration
```

## Quick Reference

```bash
# Most Common Commands

# Run all tests
pytest

# Run integration tests
pytest tests/integration

# Specific test
pytest tests/integration/test_semantic_search_integration.py::test_semantic_search_all_types

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Stop at first failure
pytest -x
```

## Additional Resources

- Pytest Documentation: https://docs.pytest.org/
- FastAPI Testing: https://fastapi.tiangolo.com/tutorial/testing/
