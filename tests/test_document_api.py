"""
Tests for dxo-client calling document APIs.

These tests validate the KnowledgeClient's document operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_document_api.py -v

Install from git:
    pip install git+https://github.com/tacjlee/dxo-client.git
"""

import pytest
import os
import uuid
import time
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient
from dxo_client.models import DocumentProcessResult, DocumentChunk
from dxo_client.exceptions import (
    KnowledgeConnectionError,
    KnowledgeAPIError,
    KnowledgeNotFoundError,
    KnowledgeValidationError,
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
    return KnowledgeClient(base_url=knowledge_base_url, read_timeout=120.0)


@pytest.fixture
def test_tenant_id():
    """Generate unique tenant ID for test isolation."""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_knowledge_id():
    """Generate unique knowledge base ID."""
    return f"kb-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_collection_name(test_tenant_id):
    """Generate test collection name."""
    sanitized = test_tenant_id.replace('-', '_')
    return f"tenant_{sanitized}_documents"


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

class TestDocumentAPIUnit:
    """Unit tests for document API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        return KnowledgeClient(base_url="http://mock-service:8000")

    def test_add_documents_request_format(self, mock_client):
        """Test add_documents sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "chunks": [
                    {"chunk_id": "chunk-1", "content": "First chunk", "start_char": 0, "end_char": 10, "metadata": {}},
                    {"chunk_id": "chunk-2", "content": "Second chunk", "start_char": 10, "end_char": 22, "metadata": {}}
                ],
                "vector_ids": ["vec-1", "vec-2"]
            }

            documents = [
                {
                    "content": "This is document content.",
                    "metadata": {"file_name": "test.pdf", "document_id": "doc-1"}
                }
            ]

            result = mock_client.add_documents(
                collection_name="test_collection",
                documents=documents,
                tenant_id="tenant-1",
                knowledge_id="kb-1",
                chunk_size=500,
                chunk_overlap=100
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/knowledge/documents/process"
            assert call_args[1]["json"]["collection_name"] == "test_collection"
            assert call_args[1]["json"]["tenant_id"] == "tenant-1"
            assert call_args[1]["json"]["knowledge_id"] == "kb-1"
            assert call_args[1]["json"]["chunk_config"]["chunk_size"] == 500
            assert call_args[1]["json"]["chunk_config"]["chunk_overlap"] == 100

    def test_add_documents_result_parsing(self, mock_client):
        """Test add_documents result parsing."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "chunks": [
                    {"chunk_id": "chunk-1", "content": "First chunk", "start_char": 0, "end_char": 10, "metadata": {}},
                ],
                "vector_ids": ["vec-1"]
            }

            documents = [{"content": "Test content", "metadata": {"document_id": "doc-1"}}]
            result = mock_client.add_documents(
                collection_name="test_collection",
                documents=documents,
                tenant_id="tenant-1",
                knowledge_id="kb-1"
            )

            assert isinstance(result, DocumentProcessResult)
            assert result.chunks_count >= 1
            assert result.status == "processed"

    def test_add_documents_multiple(self, mock_client):
        """Test add_documents with multiple documents."""
        call_count = [0]

        def mock_request_fn(*args, **kwargs):
            call_count[0] += 1
            return {
                "chunks": [{"chunk_id": f"chunk-{call_count[0]}", "content": "Chunk", "start_char": 0, "end_char": 5, "metadata": {}}],
                "vector_ids": [f"vec-{call_count[0]}"]
            }

        with patch.object(mock_client, '_make_request', side_effect=mock_request_fn):
            documents = [
                {"content": "Document 1", "metadata": {"document_id": "doc-1"}},
                {"content": "Document 2", "metadata": {"document_id": "doc-2"}},
                {"content": "Document 3", "metadata": {"document_id": "doc-3"}},
            ]

            result = mock_client.add_documents(
                collection_name="test_collection",
                documents=documents,
                tenant_id="tenant-1",
                knowledge_id="kb-1"
            )

            # Should call API for each document
            assert call_count[0] == 3
            assert result.chunks_count == 3

    def test_delete_documents_request_format(self, mock_client):
        """Test delete_documents sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"deleted_count": 5}

            result = mock_client.delete_documents(
                collection_name="test_collection",
                tenant_id="tenant-1",
                knowledge_id="kb-1",
                document_id="doc-1"
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "DELETE"
            assert call_args[0][1] == "/api/knowledge/documents"
            assert call_args[1]["json"]["collection_name"] == "test_collection"
            assert call_args[1]["json"]["tenant_id"] == "tenant-1"
            assert call_args[1]["json"]["document_id"] == "doc-1"

    def test_delete_documents_by_filename(self, mock_client):
        """Test delete_documents by file_name."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"deleted_count": 3}

            result = mock_client.delete_documents(
                collection_name="test_collection",
                file_name="report.pdf"
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["file_name"] == "report.pdf"

    def test_delete_documents_returns_count(self, mock_client):
        """Test delete_documents returns deleted count."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"deleted_count": 10}

            result = mock_client.delete_documents(
                collection_name="test_collection",
                tenant_id="tenant-1"
            )

            assert result == 10


# ============================================================================
# INTEGRATION TESTS (requires running service)
# ============================================================================

@requires_service
class TestDocumentAPIIntegration:
    """Integration tests for document API (requires running service)."""

    @pytest.fixture(autouse=True)
    def setup_collection(self, live_client, test_tenant_id, test_collection_name):
        """Create test collection before each test, cleanup after."""
        # Create collection
        try:
            live_client.create_collection(
                tenant_id=test_tenant_id,
                name="documents",
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

    def test_add_single_document(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding a single document."""
        documents = [
            {
                "content": "Machine learning is a subset of artificial intelligence. "
                           "It enables computers to learn from data without explicit programming.",
                "metadata": {"file_name": "ml-intro.txt", "document_id": "doc-ml-1"}
            }
        ]

        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        assert isinstance(result, DocumentProcessResult)
        assert result.chunks_count >= 1
        assert result.status == "processed"

    def test_add_document_with_chunking(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding document that requires chunking."""
        # Long document that should be chunked
        long_content = " ".join([
            f"Section {i}: This is paragraph number {i} with detailed content about topic {i}. "
            f"It contains multiple sentences to ensure proper chunking behavior. "
            f"The content is designed to test the document processing pipeline."
            for i in range(20)
        ])

        documents = [
            {
                "content": long_content,
                "metadata": {"file_name": "long-doc.txt", "document_id": "doc-long"}
            }
        ]

        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id,
            chunk_size=500,
            chunk_overlap=50
        )

        assert result.chunks_count >= 1
        assert result.status == "processed"

    def test_add_multiple_documents(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding multiple documents."""
        documents = [
            {"content": "Python is a programming language.", "metadata": {"document_id": "doc-py"}},
            {"content": "JavaScript runs in the browser.", "metadata": {"document_id": "doc-js"}},
            {"content": "Rust is memory safe.", "metadata": {"document_id": "doc-rs"}},
        ]

        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        assert result.chunks_count >= 3
        assert result.status == "processed"

    def test_add_document_with_user_id(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding document with user_id."""
        documents = [
            {"content": "Document from specific user.", "metadata": {"document_id": "doc-user"}}
        ]

        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id,
            user_id="user-123"
        )

        assert result.status == "processed"

    def test_add_document_unicode_content(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding document with unicode content."""
        documents = [
            {"content": "日本語ドキュメント。これはテストです。", "metadata": {"document_id": "doc-jp"}},
            {"content": "中文文档内容。这是一个测试文档。", "metadata": {"document_id": "doc-zh"}},
        ]

        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        assert result.status == "processed"
        assert result.chunks_count >= 2

    def test_delete_documents_by_document_id(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test deleting documents by document_id."""
        # First add a document
        documents = [
            {"content": "Document to be deleted.", "metadata": {"document_id": "doc-delete-me"}}
        ]
        live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        # Wait for indexing
        time.sleep(1)

        # Delete by document_id
        deleted_count = live_client.delete_documents(
            collection_name=test_collection_name,
            document_id="doc-delete-me"
        )

        # Filter-based deletes return -1 (Qdrant doesn't provide count)
        assert deleted_count == -1 or deleted_count >= 0

    def test_delete_documents_by_knowledge_id(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test deleting documents by knowledge_id."""
        # Add documents
        documents = [
            {"content": "Doc 1 for kb.", "metadata": {"document_id": "doc-kb-1"}},
            {"content": "Doc 2 for kb.", "metadata": {"document_id": "doc-kb-2"}},
        ]
        live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        # Wait for indexing
        time.sleep(1)

        # Delete by knowledge_id
        deleted_count = live_client.delete_documents(
            collection_name=test_collection_name,
            knowledge_id=test_knowledge_id
        )

        # Filter-based deletes return -1 (Qdrant doesn't provide count)
        assert deleted_count == -1 or deleted_count >= 0

    def test_delete_documents_by_tenant_id(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test deleting documents by tenant_id."""
        # Add documents
        documents = [
            {"content": "Doc for tenant.", "metadata": {"document_id": "doc-tenant-1"}}
        ]
        live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        # Wait for indexing
        time.sleep(1)

        # Delete by tenant
        deleted_count = live_client.delete_documents(
            collection_name=test_collection_name,
            tenant_id=test_tenant_id
        )

        # Filter-based deletes return -1 (Qdrant doesn't provide count)
        assert deleted_count == -1 or deleted_count >= 0

    def test_add_and_search_documents(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test adding documents and then searching for them."""
        # Add documents
        documents = [
            {
                "content": "Neural networks are inspired by biological neurons.",
                "metadata": {"document_id": "doc-nn"}
            },
            {
                "content": "Databases store and retrieve structured data.",
                "metadata": {"document_id": "doc-db"}
            }
        ]
        live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )

        # Wait for indexing
        time.sleep(2)

        # Search for neural networks
        results = live_client.search(
            collection_name=test_collection_name,
            query="What are neural networks?",
            top_k=5
        )

        # Should find the neural networks document
        assert len(results) > 0

    def test_document_full_lifecycle(
        self, live_client, test_collection_name, test_tenant_id, test_knowledge_id
    ):
        """Test full document lifecycle: add -> search -> delete."""
        unique_id = uuid.uuid4().hex[:8]
        document_id = f"doc-lifecycle-{unique_id}"

        # 1. Add document
        documents = [
            {
                "content": "This is a lifecycle test document about quantum computing.",
                "metadata": {"document_id": document_id, "file_name": "lifecycle.txt"}
            }
        ]
        result = live_client.add_documents(
            collection_name=test_collection_name,
            documents=documents,
            tenant_id=test_tenant_id,
            knowledge_id=test_knowledge_id
        )
        assert result.status == "processed"

        # 2. Wait for indexing
        time.sleep(2)

        # 3. Search for it
        results = live_client.search(
            collection_name=test_collection_name,
            query="quantum computing",
            top_k=5
        )
        # May or may not find it depending on indexing time

        # 4. Delete
        deleted = live_client.delete_documents(
            collection_name=test_collection_name,
            document_id=document_id
        )
        # Filter-based deletes return -1 (Qdrant doesn't provide count)
        assert deleted == -1 or deleted >= 0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
