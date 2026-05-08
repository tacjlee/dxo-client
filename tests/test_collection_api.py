"""
Tests for dxo-client calling collection APIs.

These tests validate the KnowledgeClient's collection operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_collection_api.py -v

Install from git:
    pip install git+https://github.com/tacjlee/dxo-client.git
"""

import pytest
import os
import uuid
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient
from dxo_client.models import CollectionInfo
from dxo_client.exceptions import (
    KnowledgeConnectionError,
    KnowledgeAPIError,
    KnowledgeNotFoundError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def knowledge_base_url():
    """Get knowledge base service URL from environment or default."""
    return os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")


@pytest.fixture
def live_client(knowledge_base_url):
    """Create a client for integration tests."""
    return KnowledgeClient(base_url=knowledge_base_url, read_timeout=60.0)


@pytest.fixture
def test_tenant_id():
    """Generate unique tenant ID for test isolation."""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"


def service_available(url: str) -> bool:
    """Check if the knowledge base service is available."""
    try:
        import httpx
        response = httpx.get(f"{url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


requires_service = pytest.mark.skipif(
    not service_available(os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")),
    reason="Knowledge base service not available"
)


# ============================================================================
# UNIT TESTS (mocked)
# ============================================================================

class TestCollectionAPIUnit:
    """Unit tests for collection API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        return KnowledgeClient(base_url="http://mock-service:8000")

    def test_create_collection_request_format(self, mock_client):
        """Test create_collection sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "name": "tenant_test_tenant_mycollection",
                "vectors_count": 0,
                "points_count": 0,
                "status": "green",
                "vector_size": 1024,
                "distance": "Cosine",
                "multivector_enabled": True
            }

            result = mock_client.create_collection(
                tenant_id="test-tenant",
                name="mycollection",
                enable_multivector=True,
                vector_size=1024
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/knowledge/collections"
            assert call_args[1]["json"]["tenant_id"] == "test-tenant"
            assert call_args[1]["json"]["name"] == "mycollection"
            assert call_args[1]["json"]["enable_multivector"] is True
            assert call_args[1]["json"]["vector_size"] == 1024

    def test_create_collection_result_parsing(self, mock_client):
        """Test create_collection result parsing."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "name": "tenant_test_tenant_mycollection",
                "vectors_count": 0,
                "status": "green",
                "config": {
                    "vector_size": 1024,
                    "distance": "Cosine"
                }
            }

            result = mock_client.create_collection("test-tenant", "mycollection")

            assert isinstance(result, CollectionInfo)
            assert result.name == "tenant_test_tenant_mycollection"
            assert result.config["vector_size"] == 1024

    def test_get_collection_info_request_format(self, mock_client):
        """Test get_collection_info sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "name": "test_collection",
                "vectors_count": 100,
                "points_count": 100,
                "status": "green",
                "vector_size": 1024,
                "distance": "Cosine",
                "multivector_enabled": True
            }

            result = mock_client.get_collection_info("test_collection")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "/api/knowledge/collections/test_collection"

    def test_list_collections_request_format(self, mock_client):
        """Test list_collections sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "collections": [
                    {"name": "coll1", "vectors_count": 10, "points_count": 10, "status": "green"},
                    {"name": "coll2", "vectors_count": 20, "points_count": 20, "status": "green"}
                ]
            }

            result = mock_client.list_collections(tenant_id="tenant-1")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "/api/knowledge/collections"
            assert call_args[1]["params"]["tenant_id"] == "tenant-1"

    def test_list_collections_result_parsing(self, mock_client):
        """Test list_collections result parsing."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "collections": [
                    {"name": "coll1", "vectors_count": 10, "points_count": 10, "status": "green"},
                    {"name": "coll2", "vectors_count": 20, "points_count": 20, "status": "green"}
                ]
            }

            result = mock_client.list_collections()

            assert len(result) == 2
            assert all(isinstance(c, CollectionInfo) for c in result)
            assert result[0].name == "coll1"
            assert result[1].vectors_count == 20

    def test_delete_collection_request_format(self, mock_client):
        """Test delete_collection sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"status": "deleted"}

            result = mock_client.delete_collection(
                collection_name="test_collection",
                tenant_id="tenant-1",
                force=True
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "DELETE"
            assert call_args[0][1] == "/api/knowledge/collections/test_collection"
            assert call_args[1]["params"]["tenant_id"] == "tenant-1"
            assert call_args[1]["params"]["force"] is True

    def test_delete_collection_returns_true(self, mock_client):
        """Test delete_collection returns True on success."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"status": "deleted"}

            result = mock_client.delete_collection("test_collection")

            assert result is True


# ============================================================================
# INTEGRATION TESTS (requires running service)
# ============================================================================

@requires_service
class TestCollectionAPIIntegration:
    """Integration tests for collection API (requires running service)."""

    def test_create_collection_basic(self, live_client, test_tenant_id):
        """Test creating a basic collection."""
        collection_name = "basic"

        try:
            result = live_client.create_collection(
                tenant_id=test_tenant_id,
                name=collection_name
            )

            assert isinstance(result, CollectionInfo)
            assert test_tenant_id.replace('-', '_') in result.name
            assert result.status in ("green", "yellow")
        finally:
            # Cleanup
            sanitized = test_tenant_id.replace('-', '_')
            full_name = f"tenant_{sanitized}_{collection_name}"
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_create_collection_with_multivector(self, live_client, test_tenant_id):
        """Test creating a multivector collection."""
        collection_name = "multivec"

        try:
            result = live_client.create_collection(
                tenant_id=test_tenant_id,
                name=collection_name,
                enable_multivector=True,
                vector_size=1024
            )

            assert isinstance(result, CollectionInfo)
            assert result.config is not None
            assert result.config.get("vector_size") == 1024
        finally:
            sanitized = test_tenant_id.replace('-', '_')
            full_name = f"tenant_{sanitized}_{collection_name}"
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_create_collection_dense_only(self, live_client, test_tenant_id):
        """Test creating a dense-only collection."""
        collection_name = "dense"

        try:
            result = live_client.create_collection(
                tenant_id=test_tenant_id,
                name=collection_name,
                enable_multivector=False,
                vector_size=1024
            )

            assert isinstance(result, CollectionInfo)
        finally:
            sanitized = test_tenant_id.replace('-', '_')
            full_name = f"tenant_{sanitized}_{collection_name}"
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_get_collection_info(self, live_client, test_tenant_id):
        """Test getting collection info."""
        collection_name = "getinfo"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        try:
            # Create collection first
            live_client.create_collection(test_tenant_id, collection_name)

            # Get info
            info = live_client.get_collection_info(full_name)

            assert isinstance(info, CollectionInfo)
            assert info.name == full_name
        finally:
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_get_collection_info_not_found(self, live_client):
        """Test getting info for non-existent collection."""
        with pytest.raises(KnowledgeNotFoundError):
            live_client.get_collection_info("nonexistent_collection_xyz123")

    def test_list_collections_all(self, live_client, test_tenant_id):
        """Test listing all collections."""
        collection_name = "listall"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        try:
            # Create a test collection
            live_client.create_collection(test_tenant_id, collection_name)

            # List all collections
            collections = live_client.list_collections()

            assert isinstance(collections, list)
            assert len(collections) >= 1
            # Our collection should be in the list
            names = [c.name for c in collections]
            assert full_name in names
        finally:
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_list_collections_by_tenant(self, live_client, test_tenant_id):
        """Test listing collections filtered by tenant."""
        collection_name = "listtenant"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        try:
            # Create a test collection
            live_client.create_collection(test_tenant_id, collection_name)

            # List by tenant
            collections = live_client.list_collections(tenant_id=test_tenant_id)

            assert isinstance(collections, list)
            # Should only include collections for this tenant
            for c in collections:
                assert sanitized in c.name
        finally:
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_delete_collection(self, live_client, test_tenant_id):
        """Test deleting a collection."""
        collection_name = "todelete"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        # Create collection
        live_client.create_collection(test_tenant_id, collection_name)

        # Delete it
        result = live_client.delete_collection(full_name, force=True)
        assert result is True

        # Verify it's gone
        with pytest.raises(KnowledgeNotFoundError):
            live_client.get_collection_info(full_name)

    def test_delete_collection_with_tenant(self, live_client, test_tenant_id):
        """Test deleting a collection with tenant ID."""
        collection_name = "delwithtenant"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        # Create collection
        live_client.create_collection(test_tenant_id, collection_name)

        # Delete with tenant
        result = live_client.delete_collection(
            full_name,
            tenant_id=test_tenant_id,
            force=True
        )
        assert result is True

    def test_create_duplicate_collection(self, live_client, test_tenant_id):
        """Test creating duplicate collection is idempotent (returns existing)."""
        collection_name = "duplicate"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        try:
            # Create first time
            result1 = live_client.create_collection(test_tenant_id, collection_name)

            # Create again - should return existing collection (idempotent)
            result2 = live_client.create_collection(test_tenant_id, collection_name)

            # Both should have the same name
            assert result1.name == result2.name == full_name
        finally:
            try:
                live_client.delete_collection(full_name, force=True)
            except Exception:
                pass

    def test_collection_lifecycle(self, live_client, test_tenant_id):
        """Test full collection lifecycle: create -> get -> list -> delete."""
        collection_name = "lifecycle"
        sanitized = test_tenant_id.replace('-', '_')
        full_name = f"tenant_{sanitized}_{collection_name}"

        # 1. Create
        created = live_client.create_collection(
            test_tenant_id,
            collection_name,
            enable_multivector=True
        )
        assert created.name == full_name

        # 2. Get info
        info = live_client.get_collection_info(full_name)
        assert info.name == full_name
        assert info.vectors_count == 0

        # 3. List and verify included
        collections = live_client.list_collections(tenant_id=test_tenant_id)
        names = [c.name for c in collections]
        assert full_name in names

        # 4. Delete
        deleted = live_client.delete_collection(full_name, force=True)
        assert deleted is True

        # 5. Verify gone
        with pytest.raises(KnowledgeNotFoundError):
            live_client.get_collection_info(full_name)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
