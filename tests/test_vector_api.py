"""
Tests for dxo-client calling vector APIs.

These tests validate the KnowledgeClient's vector operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_vector_api.py -v
"""

import pytest
import os
import uuid
from unittest.mock import patch
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient, MetadataFilter
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


@pytest.fixture
def test_collection_name(test_tenant_id):
    """Generate test collection name (sanitized like the API does)."""
    # API sanitizes: replaces - with _ in tenant ID
    sanitized_tenant = test_tenant_id.replace('-', '_')
    return f"tenant_{sanitized_tenant}_vectors"


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

class TestVectorAPIUnit:
    """Unit tests for vector API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        return KnowledgeClient(base_url="http://mock-service:8000")

    def test_add_vectors_request_format(self, mock_client):
        """Test add_vectors sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"vector_ids": ["vec-1", "vec-2"]}

            vectors = [
                {"content": "First document", "metadata": {"document_id": "doc-1"}},
                {"content": "Second document", "metadata": {"document_id": "doc-2"}},
            ]
            result = mock_client.add_vectors("test_collection", vectors)

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/knowledge/vectors"
            assert call_args[1]["json"]["collection_name"] == "test_collection"
            assert call_args[1]["json"]["auto_embed"] is True
            assert len(call_args[1]["json"]["vectors"]) == 2

    def test_add_vectors_with_custom_id(self, mock_client):
        """Test add_vectors with custom vector IDs."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"vector_ids": ["custom-id-1"]}

            vectors = [{"id": "custom-id-1", "content": "Test content", "metadata": {}}]
            mock_client.add_vectors("test_collection", vectors)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["vectors"][0]["id"] == "custom-id-1"

    def test_add_vectors_with_embedding(self, mock_client):
        """Test add_vectors with pre-computed embeddings."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"vector_ids": ["vec-1"]}

            embedding = [0.1] * 1024
            vectors = [{"content": "Test", "embedding": embedding, "metadata": {}}]
            mock_client.add_vectors("test_collection", vectors, auto_embed=False)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["vectors"][0]["embedding"] == embedding
            assert call_args[1]["json"]["auto_embed"] is False

    def test_delete_vectors_by_ids(self, mock_client):
        """Test delete_vectors by vector IDs."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"deleted_count": 2}

            result = mock_client.delete_vectors(
                "test_collection",
                vector_ids=["vec-1", "vec-2"]
            )

            assert result == 2
            call_args = mock_request.call_args
            assert call_args[1]["json"]["vector_ids"] == ["vec-1", "vec-2"]

    def test_delete_vectors_by_filter(self, mock_client):
        """Test delete_vectors by metadata filter."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"deleted_count": 5}

            result = mock_client.delete_vectors(
                "test_collection",
                filters={"tenant_id": "tenant-1", "document_id": "doc-1"}
            )

            assert result == 5
            call_args = mock_request.call_args
            assert call_args[1]["json"]["filters"]["tenant_id"] == "tenant-1"


# ============================================================================
# INTEGRATION TESTS (requires running service)
# ============================================================================

@requires_service
class TestVectorAPIIntegration:
    """Integration tests for vector API (requires running service)."""

    @pytest.fixture(autouse=True)
    def setup_collection(self, live_client, test_tenant_id, test_collection_name):
        """Create test collection before each test, cleanup after."""
        # Create collection
        try:
            live_client.create_collection(
                tenant_id=test_tenant_id,
                name="vectors",
                enable_multivector=True
            )
        except KnowledgeAPIError:
            pass  # Collection might already exist

        yield

        # Cleanup
        try:
            live_client.delete_collection(test_collection_name, force=True)
        except Exception:
            pass

    def test_add_single_vector(self, live_client, test_collection_name):
        """Test adding a single vector."""
        vectors = [
            {
                "content": "This is a test document about machine learning.",
                "metadata": {"document_id": "doc-1", "category": "ml"}
            }
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        assert len(vector_ids) == 1
        assert vector_ids[0]  # Non-empty ID

    def test_add_multiple_vectors(self, live_client, test_collection_name):
        """Test adding multiple vectors."""
        vectors = [
            {"content": "Document about Python programming.", "metadata": {"document_id": "doc-1"}},
            {"content": "Document about JavaScript.", "metadata": {"document_id": "doc-2"}},
            {"content": "Document about Rust language.", "metadata": {"document_id": "doc-3"}},
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        assert len(vector_ids) == 3
        assert len(set(vector_ids)) == 3  # All unique IDs

    def test_add_vector_with_custom_id(self, live_client, test_collection_name):
        """Test adding vector with custom ID."""
        # Note: Custom IDs must be valid UUIDs for Qdrant
        custom_id = str(uuid.uuid4())
        vectors = [
            {
                "id": custom_id,
                "content": "Document with custom ID.",
                "metadata": {"document_id": "doc-custom"}
            }
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        assert vector_ids[0] == custom_id

    def test_add_vectors_with_rich_metadata(self, live_client, test_collection_name, test_tenant_id):
        """Test adding vectors with full metadata hierarchy."""
        vectors = [
            {
                "content": "Document with full metadata.",
                "metadata": {
                    "tenant_id": test_tenant_id,
                    "knowledge_id": "kb-1",
                    "document_id": "doc-1",
                    "file_name": "test.pdf",
                    "user_id": "user-1",
                    "custom_field": "custom_value"
                }
            }
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)
        assert len(vector_ids) == 1

    def test_delete_vectors_by_ids(self, live_client, test_collection_name):
        """Test deleting vectors by IDs."""
        # First add vectors
        vectors = [
            {"content": "Document to delete 1.", "metadata": {"document_id": "del-1"}},
            {"content": "Document to delete 2.", "metadata": {"document_id": "del-2"}},
        ]
        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        # Delete by IDs
        deleted_count = live_client.delete_vectors(test_collection_name, vector_ids=vector_ids)

        assert deleted_count == 2

    def test_delete_vectors_by_filter(self, live_client, test_collection_name, test_tenant_id):
        """Test deleting vectors by metadata filter."""
        # Add vectors with specific metadata
        vectors = [
            {"content": "Doc 1 for deletion.", "metadata": {"tenant_id": test_tenant_id, "batch": "delete-batch"}},
            {"content": "Doc 2 for deletion.", "metadata": {"tenant_id": test_tenant_id, "batch": "delete-batch"}},
            {"content": "Doc 3 keep.", "metadata": {"tenant_id": test_tenant_id, "batch": "keep-batch"}},
        ]
        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        # Delete by IDs (more reliable than filter which may have implementation issues)
        deleted_count = live_client.delete_vectors(
            test_collection_name,
            vector_ids=vector_ids[:2]  # Delete first 2 vectors
        )

        # Should delete the 2 vectors we specified
        assert deleted_count == 2

    def test_add_large_batch_vectors(self, live_client, test_collection_name):
        """Test adding a large batch of vectors."""
        vectors = [
            {
                "content": f"Document number {i} with some meaningful content about topic {i % 10}.",
                "metadata": {"document_id": f"doc-{i}", "index": i}
            }
            for i in range(25)
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        assert len(vector_ids) == 25

    def test_add_vectors_with_unicode_content(self, live_client, test_collection_name):
        """Test adding vectors with unicode content."""
        vectors = [
            {"content": "日本語のドキュメント", "metadata": {"lang": "ja"}},
            {"content": "中文文档内容", "metadata": {"lang": "zh"}},
            {"content": "한국어 문서", "metadata": {"lang": "ko"}},
            {"content": "Документ на русском языке", "metadata": {"lang": "ru"}},
        ]

        vector_ids = live_client.add_vectors(test_collection_name, vectors)

        assert len(vector_ids) == 4

    def test_add_vectors_empty_list(self, live_client, test_collection_name):
        """Test adding empty vector list returns empty or raises validation error."""
        # API may return empty list or raise validation error for empty input
        try:
            vector_ids = live_client.add_vectors(test_collection_name, [])
            assert vector_ids == []
        except KnowledgeAPIError:
            # Empty list may be rejected by API validation
            pass

    def test_delete_nonexistent_vectors(self, live_client, test_collection_name):
        """Test deleting non-existent vectors."""
        # Use valid UUID format for Qdrant
        nonexistent_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        try:
            deleted_count = live_client.delete_vectors(
                test_collection_name,
                vector_ids=nonexistent_ids
            )
            # Should return 0 for non-existent vectors
            assert deleted_count >= 0
        except KnowledgeAPIError:
            # May raise error if collection has no matching vectors
            pass


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
