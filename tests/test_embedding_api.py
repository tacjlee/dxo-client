"""
Tests for dxo-client calling embedding_service APIs.

These tests validate the KnowledgeClient's embedding operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_embedding_api.py -v

Requirements:
    - workflow-knowledge service running at http://localhost:8010
    - Or set KNOWLEDGE_BASE_SERVICE_URL environment variable
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from typing import List

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient
from dxo_client.exceptions import (
    KnowledgeConnectionError,
    KnowledgeTimeoutError,
    KnowledgeAPIError,
)


# ============================================================================
# UNIT TESTS (mocked - no service required)
# ============================================================================

class TestEmbeddingAPIUnit:
    """Unit tests for embedding API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        client = KnowledgeClient(base_url="http://mock-service:8000")
        return client

    def test_generate_embeddings_request_format(self, mock_client):
        """Test that generate_embeddings sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "embeddings": [[0.1] * 1024, [0.2] * 1024],
                "model": "bge-m3-onnx",
                "dimension": 1024
            }

            texts = ["Hello world", "Test document"]
            result = mock_client.generate_embeddings(texts)

            # Verify request was made with correct params
            mock_request.assert_called_once_with(
                "POST",
                "/api/knowledge/embeddings",
                json={
                    "texts": texts,
                    "batch_size": 32,
                    "use_cache": True
                }
            )

            # Verify response parsing
            assert len(result) == 2
            assert len(result[0]) == 1024

    def test_generate_embeddings_empty_list(self, mock_client):
        """Test generate_embeddings with empty list."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"embeddings": []}

            result = mock_client.generate_embeddings([])

            assert result == []

    def test_generate_embeddings_custom_batch_size(self, mock_client):
        """Test generate_embeddings with custom batch size."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {"embeddings": [[0.1] * 1024]}

            mock_client.generate_embeddings(["test"], batch_size=16)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["batch_size"] == 16

    def test_generate_embeddings_connection_error(self, mock_client):
        """Test generate_embeddings handles connection errors."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeConnectionError("Connection refused")

            with pytest.raises(KnowledgeConnectionError):
                mock_client.generate_embeddings(["test"])

    def test_generate_embeddings_timeout_error(self, mock_client):
        """Test generate_embeddings handles timeout errors."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeTimeoutError("Request timed out")

            with pytest.raises(KnowledgeTimeoutError):
                mock_client.generate_embeddings(["test"])

    def test_generate_embeddings_api_error(self, mock_client):
        """Test generate_embeddings handles API errors."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeAPIError(
                "Server error",
                status_code=500,
                response_body="Internal error"
            )

            with pytest.raises(KnowledgeAPIError):
                mock_client.generate_embeddings(["test"])


class TestEmbeddingAPIRetry:
    """Test retry behavior for embedding API."""

    def test_retry_on_connection_error(self):
        """Test that connection errors trigger retries."""
        client = KnowledgeClient(base_url="http://mock-service:8000", max_retries=3)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise KnowledgeConnectionError("Connection failed")
            return {"embeddings": [[0.1] * 1024]}

        with patch.object(client, '_make_request', side_effect=mock_request):
            # Should succeed after retries
            result = client.generate_embeddings(["test"])
            assert len(result) == 1
            assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that exceeding max retries raises exception."""
        client = KnowledgeClient(base_url="http://mock-service:8000", max_retries=2)

        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeConnectionError("Connection failed")

            with pytest.raises(KnowledgeConnectionError):
                client.generate_embeddings(["test"])


# ============================================================================
# INTEGRATION TESTS (requires running service)
# ============================================================================

@pytest.fixture
def knowledge_base_url():
    """Get knowledge base service URL from environment or default."""
    return os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")


@pytest.fixture
def live_client(knowledge_base_url):
    """Create a client for integration tests."""
    return KnowledgeClient(base_url=knowledge_base_url, read_timeout=60.0)


def service_available(url: str) -> bool:
    """Check if the knowledge base service is available."""
    try:
        import httpx
        response = httpx.get(f"{url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


# Mark integration tests to skip if service unavailable
requires_service = pytest.mark.skipif(
    not service_available(os.environ.get("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8010")),
    reason="Knowledge base service not available"
)


@requires_service
class TestEmbeddingAPIIntegration:
    """Integration tests for embedding API (requires running service)."""

    def test_health_check(self, live_client):
        """Test service health check."""
        health = live_client.health_check()
        assert health["status"] == "healthy"

    def test_generate_single_embedding(self, live_client):
        """Test generating embedding for single text."""
        texts = ["Hello, this is a test document for embedding generation."]
        embeddings = live_client.generate_embeddings(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1024  # BGE-M3 dimension
        assert all(isinstance(v, float) for v in embeddings[0])

    def test_generate_batch_embeddings(self, live_client):
        """Test generating embeddings for multiple texts."""
        texts = [
            "First document about machine learning.",
            "Second document about natural language processing.",
            "Third document about vector databases.",
        ]
        embeddings = live_client.generate_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1024

    def test_embedding_consistency(self, live_client):
        """Test that same text produces same embedding (via cache)."""
        text = "Consistent embedding test document."

        emb1 = live_client.generate_embeddings([text])[0]
        emb2 = live_client.generate_embeddings([text])[0]

        # Should be identical (cached)
        assert emb1 == emb2

    def test_embedding_similarity(self, live_client):
        """Test that similar texts have similar embeddings."""
        texts = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat.",
            "Python is a programming language.",
        ]
        embeddings = live_client.generate_embeddings(texts)

        # Calculate cosine similarity
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        # Similar sentences should have higher similarity
        sim_similar = cosine_similarity(embeddings[0], embeddings[1])
        sim_different = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_similar > sim_different
        assert sim_similar > 0.8  # Similar sentences
        assert sim_different < 0.7  # Different topics

    def test_large_batch_embeddings(self, live_client):
        """Test generating embeddings for large batch."""
        texts = [f"Document number {i} with some content." for i in range(50)]
        embeddings = live_client.generate_embeddings(texts, batch_size=16)

        assert len(embeddings) == 50
        for emb in embeddings:
            assert len(emb) == 1024

    def test_unicode_text_embedding(self, live_client):
        """Test embedding generation for unicode text."""
        texts = [
            "日本語のテキスト",  # Japanese
            "中文文本",  # Chinese
            "한국어 텍스트",  # Korean
            "Текст на русском",  # Russian
            "Emoji test 🎉🚀💻",
        ]
        embeddings = live_client.generate_embeddings(texts)

        assert len(embeddings) == 5
        for emb in embeddings:
            assert len(emb) == 1024

    def test_long_text_embedding(self, live_client):
        """Test embedding generation for long text."""
        # BGE-M3 has 8192 token limit
        long_text = "This is a test sentence. " * 500  # ~3000 words
        embeddings = live_client.generate_embeddings([long_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1024

    def test_empty_string_embedding(self, live_client):
        """Test embedding generation for empty string."""
        texts = ["", "Non-empty text"]
        embeddings = live_client.generate_embeddings(texts)

        assert len(embeddings) == 2
        # Empty string should still produce an embedding
        assert len(embeddings[0]) == 1024


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run unit tests by default, integration tests if service available
    pytest.main([__file__, "-v", "--tb=short"])
