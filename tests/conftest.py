"""
Pytest configuration and shared fixtures for dxo-client tests.

Install dxo-client from git:
    pip install git+https://github.com/tacjlee/dxo-client.git

Or for development:
    pip install -e .

Set environment variable for service URL:
    export KNOWLEDGE_BASE_SERVICE_URL=http://localhost:8010
"""

import pytest
import os
import uuid

# Set default test environment
os.environ.setdefault("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def knowledge_base_url():
    """Get knowledge base service URL from environment."""
    return os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")


@pytest.fixture
def test_tenant_id():
    """Generate unique tenant ID for test isolation."""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_knowledge_id():
    """Generate unique knowledge base ID."""
    return f"kb-{uuid.uuid4().hex[:8]}"


# ============================================================================
# SERVICE AVAILABILITY CHECK
# ============================================================================

def service_available(url: str) -> bool:
    """Check if the knowledge base service is available."""
    try:
        import httpx
        response = httpx.get(f"{url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip marker for integration tests
requires_service = pytest.mark.skipif(
    not service_available(os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")),
    reason="Knowledge base service not available"
)


# ============================================================================
# TEST SESSION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring running service"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to tests in Integration classes
        if "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# ============================================================================
# TEST REPORTING
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def report_service_status():
    """Report service status at start of test session."""
    url = os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")
    available = service_available(url)
    print(f"\n{'=' * 60}")
    print(f"Knowledge Base Service: {url}")
    print(f"Status: {'AVAILABLE' if available else 'UNAVAILABLE'}")
    if not available:
        print("NOTE: Integration tests will be skipped")
    print(f"{'=' * 60}\n")
    yield
