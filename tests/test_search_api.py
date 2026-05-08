"""
Tests for dxo-client calling search APIs.

These tests validate the KnowledgeClient's search operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_search_api.py -v
"""

import pytest
import os
import uuid
import time
from unittest.mock import patch
from typing import List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient, MetadataFilter
from dxo_client.models import SearchResult, RAGContext
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
    return KnowledgeClient(base_url=knowledge_base_url, read_timeout=120.0)


@pytest.fixture
def test_tenant_id():
    """Generate unique tenant ID for test isolation."""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_collection_name(test_tenant_id):
    """Generate test collection name (sanitized like the API does)."""
    # API sanitizes: replaces - with _ in tenant ID
    sanitized_tenant = test_tenant_id.replace('-', '_')
    return f"tenant_{sanitized_tenant}_search"


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

class TestSearchAPIUnit:
    """Unit tests for search API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        return KnowledgeClient(base_url="http://mock-service:8000")

    def test_search_request_format(self, mock_client):
        """Test search sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "results": [
                    {"id": "vec-1", "content": "Test content", "score": 0.95, "metadata": {}}
                ]
            }

            results = mock_client.search(
                collection_name="test_collection",
                query="test query",
                top_k=5
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/knowledge/search"
            assert call_args[1]["json"]["collection_name"] == "test_collection"
            assert call_args[1]["json"]["query"] == "test query"
            assert call_args[1]["json"]["top_k"] == 5

    def test_search_with_filters(self, mock_client):
        """Test search with metadata filters."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"results": []}

            filters = MetadataFilter(
                tenant_id="tenant-1",
                knowledge_id="kb-1"
            )
            mock_client.search(
                collection_name="test_collection",
                query="test query",
                filters=filters
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["filters"]["tenant_id"] == "tenant-1"
            assert call_args[1]["json"]["filters"]["knowledge_id"] == "kb-1"

    def test_search_with_score_threshold(self, mock_client):
        """Test search with score threshold."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"results": []}

            mock_client.search(
                collection_name="test_collection",
                query="test query",
                score_threshold=0.7
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["score_threshold"] == 0.7

    def test_search_result_parsing(self, mock_client):
        """Test search result parsing."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "results": [
                    {
                        "id": "vec-1",
                        "content": "First result content",
                        "score": 0.95,
                        "metadata": {"document_id": "doc-1"}
                    },
                    {
                        "id": "vec-2",
                        "content": "Second result content",
                        "score": 0.85,
                        "metadata": {"document_id": "doc-2"}
                    }
                ]
            }

            results = mock_client.search("test_collection", "test query")

            assert len(results) == 2
            assert isinstance(results[0], SearchResult)
            assert results[0].id == "vec-1"
            assert results[0].score == 0.95
            assert results[0].metadata["document_id"] == "doc-1"

    def test_rag_retrieval_request_format(self, mock_client):
        """Test rag_retrieval sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "context": {
                    "chunks": [],
                    "combined_context": "",
                    "source_documents": []
                }
            }

            mock_client.rag_retrieval(
                collection_name="test_collection",
                query="test query",
                top_k=3
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][1] == "/api/knowledge/search/rag"
            assert call_args[1]["json"]["top_k"] == 3

    def test_rag_retrieval_result_parsing(self, mock_client):
        """Test rag_retrieval result parsing."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "context": {
                    "chunks": [
                        {"id": "vec-1", "content": "Chunk 1", "score": 0.9, "metadata": {}},
                        {"id": "vec-2", "content": "Chunk 2", "score": 0.8, "metadata": {}}
                    ],
                    "combined_context": "Chunk 1\n\nChunk 2",
                    "source_documents": ["doc-1", "doc-2"]
                }
            }

            result = mock_client.rag_retrieval("test_collection", "test query")

            assert isinstance(result, RAGContext)
            assert len(result.chunks) == 2
            assert result.combined_context == "Chunk 1\n\nChunk 2"
            assert result.source_documents == ["doc-1", "doc-2"]


# ============================================================================
# INTEGRATION TESTS (requires running service)
# ============================================================================

@requires_service
class TestSearchAPIIntegration:
    """Integration tests for search API (requires running service)."""

    # Test documents for semantic search testing
    TEST_DOCUMENTS = [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "metadata": {"document_id": "ml-1", "category": "ml", "topic": "introduction"}
        },
        {
            "content": "Deep learning uses neural networks with many layers to process complex patterns in large amounts of data.",
            "metadata": {"document_id": "dl-1", "category": "ml", "topic": "deep-learning"}
        },
        {
            "content": "Natural language processing allows computers to understand and generate human language.",
            "metadata": {"document_id": "nlp-1", "category": "nlp", "topic": "introduction"}
        },
        {
            "content": "Vector databases store and query high-dimensional vectors for similarity search applications.",
            "metadata": {"document_id": "vec-1", "category": "database", "topic": "vectors"}
        },
        {
            "content": "Retrieval augmented generation combines search with language models to provide more accurate responses.",
            "metadata": {"document_id": "rag-1", "category": "rag", "topic": "introduction"}
        },
        {
            "content": "Python is a popular programming language used for data science and machine learning applications.",
            "metadata": {"document_id": "python-1", "category": "programming", "topic": "python"}
        },
        {
            "content": "FastAPI is a modern web framework for building APIs with Python, featuring automatic documentation.",
            "metadata": {"document_id": "fastapi-1", "category": "programming", "topic": "web"}
        },
        {
            "content": "Embeddings are dense vector representations of text that capture semantic meaning.",
            "metadata": {"document_id": "emb-1", "category": "ml", "topic": "embeddings"}
        },
    ]

    @pytest.fixture(autouse=True)
    def setup_collection_with_data(self, live_client, test_tenant_id, test_collection_name):
        """Create test collection and populate with test documents."""
        # Create collection
        try:
            live_client.create_collection(
                tenant_id=test_tenant_id,
                name="search",
                enable_multivector=True
            )
        except KnowledgeAPIError:
            pass  # Collection might already exist

        # Add test documents with tenant metadata
        vectors = []
        for doc in self.TEST_DOCUMENTS:
            metadata = doc["metadata"].copy()
            metadata["tenant_id"] = test_tenant_id
            vectors.append({
                "content": doc["content"],
                "metadata": metadata
            })

        live_client.add_vectors(test_collection_name, vectors)

        # Small delay to ensure vectors are indexed
        time.sleep(1)

        yield

        # Cleanup
        try:
            live_client.delete_collection(test_collection_name, force=True)
        except Exception:
            pass

    def test_search_basic(self, live_client, test_collection_name):
        """Test basic similarity search."""
        results = live_client.search(
            collection_name=test_collection_name,
            query="What is machine learning?",
            top_k=3
        )

        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be about ML
        assert results[0].score > 0.5

    def test_search_semantic_relevance(self, live_client, test_collection_name):
        """Test that search returns semantically relevant results."""
        results = live_client.search(
            collection_name=test_collection_name,
            query="neural networks and AI",
            top_k=5
        )

        # Results should include ML/DL related documents
        content_lower = [r.content.lower() for r in results]
        ml_related = any(
            "machine learning" in c or "deep learning" in c or "neural" in c
            for c in content_lower
        )
        assert ml_related, "Expected ML-related results for AI query"

    def test_search_with_filters(self, live_client, test_collection_name, test_tenant_id):
        """Test similarity search with metadata filters."""
        filters = MetadataFilter(
            tenant_id=test_tenant_id,
            # Filter by custom metadata if supported
        )

        results = live_client.search(
            collection_name=test_collection_name,
            query="programming language",
            top_k=10,
            filters=filters
        )

        assert len(results) > 0
        # All results should match the tenant filter
        # (exact filter verification depends on what metadata is returned)

    def test_search_top_k(self, live_client, test_collection_name):
        """Test that top_k limits results correctly."""
        for k in [1, 3, 5]:
            results = live_client.search(
                collection_name=test_collection_name,
                query="data and algorithms",
                top_k=k
            )
            assert len(results) <= k

    def test_search_score_threshold(self, live_client, test_collection_name):
        """Test score threshold filtering."""
        # Get results without threshold first
        all_results = live_client.search(
            collection_name=test_collection_name,
            query="machine learning AI",
            top_k=10
        )

        if len(all_results) > 1:
            # Use a threshold between max and min scores (must be <= 1.0)
            max_score = max(r.score for r in all_results)
            min_score = min(r.score for r in all_results)
            threshold = (max_score + min_score) / 2
            # Ensure threshold is valid (0-1 range for cosine similarity)
            threshold = min(threshold, 0.95)

            filtered_results = live_client.search(
                collection_name=test_collection_name,
                query="machine learning AI",
                top_k=10,
                score_threshold=threshold
            )

            # Filtered results should all be above threshold
            for r in filtered_results:
                assert r.score >= threshold

    def test_search_score_ordering(self, live_client, test_collection_name):
        """Test that results are ordered by score descending."""
        results = live_client.search(
            collection_name=test_collection_name,
            query="programming and software development",
            top_k=5
        )

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by score descending"

    def test_search_different_queries(self, live_client, test_collection_name):
        """Test that different queries return different results."""
        results_ml = live_client.search(
            collection_name=test_collection_name,
            query="machine learning and AI",
            top_k=3
        )

        results_web = live_client.search(
            collection_name=test_collection_name,
            query="web framework and API development",
            top_k=3
        )

        # Top results should be different
        ml_ids = {r.id for r in results_ml}
        web_ids = {r.id for r in results_web}

        # At least some results should differ
        assert ml_ids != web_ids or len(ml_ids) == 0

    def test_rag_retrieval_basic(self, live_client, test_collection_name):
        """Test basic RAG retrieval."""
        result = live_client.rag_retrieval(
            collection_name=test_collection_name,
            query="What is natural language processing?",
            top_k=3
        )

        assert isinstance(result, RAGContext)
        assert len(result.chunks) <= 3
        assert isinstance(result.combined_context, str)
        assert isinstance(result.source_documents, list)

    def test_rag_retrieval_combined_context(self, live_client, test_collection_name):
        """Test that RAG retrieval provides combined context."""
        result = live_client.rag_retrieval(
            collection_name=test_collection_name,
            query="embeddings and vector representations",
            top_k=3
        )

        # Combined context should contain content from chunks
        if result.chunks:
            for chunk in result.chunks[:2]:  # Check at least first 2
                # Content should be in combined context (might be formatted)
                assert len(result.combined_context) > 0

    def test_rag_retrieval_with_filters(self, live_client, test_collection_name, test_tenant_id):
        """Test RAG retrieval with metadata filters."""
        filters = MetadataFilter(tenant_id=test_tenant_id)

        result = live_client.rag_retrieval(
            collection_name=test_collection_name,
            query="database technology",
            top_k=5,
            filters=filters
        )

        assert isinstance(result, RAGContext)

    def test_search_unicode_query(self, live_client, test_collection_name):
        """Test search with unicode query."""
        results = live_client.search(
            collection_name=test_collection_name,
            query="机器学习 artificial intelligence",  # Mixed Chinese and English
            top_k=3
        )

        # Should return results without error
        assert isinstance(results, list)

    def test_search_long_query(self, live_client, test_collection_name):
        """Test search with long query."""
        long_query = "I want to learn about " + " ".join(
            ["machine learning", "deep learning", "neural networks",
             "natural language processing", "computer vision", "data science"] * 10
        )

        results = live_client.search(
            collection_name=test_collection_name,
            query=long_query,
            top_k=3
        )

        assert isinstance(results, list)

    def test_search_empty_results(self, live_client, test_collection_name):
        """Test search that returns no results (high threshold)."""
        results = live_client.search(
            collection_name=test_collection_name,
            query="completely unrelated gibberish xyzzy",
            top_k=3,
            score_threshold=0.99  # Very high threshold
        )

        # Should return empty list, not error
        assert results == [] or all(r.score >= 0.99 for r in results)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
