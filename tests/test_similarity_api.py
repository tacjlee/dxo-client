"""
Tests for dxo-client calling similarity APIs.

These tests validate the KnowledgeClient's similarity operations against
the workflow-knowledge service.

Run with:
    pytest tests/test_similarity_api.py -v

Requirements:
    - workflow-knowledge service running at http://localhost:8010
    - Or set KNOWLEDGE_BASE_SERVICE_URL environment variable
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import (
    KnowledgeClient,
    SimilarityResponse,
    BatchSimilarityResponse,
    BatchSimilarityResult,
)
from dxo_client.exceptions import (
    KnowledgeConnectionError,
    KnowledgeTimeoutError,
    KnowledgeAPIError,
)


# ============================================================================
# UNIT TESTS (mocked - no service required)
# ============================================================================

class TestSimilarityAPIUnit:
    """Unit tests for similarity API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        client = KnowledgeClient(base_url="http://mock-service:8000")
        return client

    def test_compute_similarity_request_format(self, mock_client):
        """Test that compute_similarity sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "similarity": 0.95,
                "model": "bge-m3-onnx",
                "execution_time_ms": 50.0
            }

            result = mock_client.compute_similarity(
                text1="Login button should be visible",
                text2="The login button must be displayed"
            )

            # Verify request was made with correct params
            mock_request.assert_called_once_with(
                "POST",
                "/api/knowledge/similarity",
                json={
                    "text1": "Login button should be visible",
                    "text2": "The login button must be displayed",
                    "use_cache": True
                }
            )

            # Verify response parsing
            assert isinstance(result, SimilarityResponse)
            assert result.similarity == 0.95
            assert result.model == "bge-m3-onnx"
            assert result.execution_time_ms == 50.0

    def test_compute_similarity_without_cache(self, mock_client):
        """Test compute_similarity with cache disabled."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "similarity": 0.80,
                "model": "bge-m3-onnx",
                "execution_time_ms": 100.0
            }

            mock_client.compute_similarity(
                text1="text1",
                text2="text2",
                use_cache=False
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["use_cache"] is False

    def test_compute_similarity_connection_error(self, mock_client):
        """Test compute_similarity handles connection errors."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeConnectionError("Connection refused")

            with pytest.raises(KnowledgeConnectionError):
                mock_client.compute_similarity("text1", "text2")

    def test_compute_similarity_timeout_error(self, mock_client):
        """Test compute_similarity handles timeout errors."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeTimeoutError("Request timed out")

            with pytest.raises(KnowledgeTimeoutError):
                mock_client.compute_similarity("text1", "text2")


class TestBatchSimilarityAPIUnit:
    """Unit tests for batch similarity API calls (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        client = KnowledgeClient(base_url="http://mock-service:8000")
        return client

    def test_compute_batch_similarity_request_format(self, mock_client):
        """Test that compute_batch_similarity sends correct request format."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "results": [
                    {"index": 0, "similarity": 0.95},
                    {"index": 1, "similarity": 0.60}
                ],
                "model": "bge-m3-onnx",
                "count": 2,
                "execution_time_ms": 75.0
            }

            pairs = [
                {"text1": "Login button", "text2": "Login button displayed"},
                {"text1": "Submit form", "text2": "Different text"}
            ]
            result = mock_client.compute_batch_similarity(pairs)

            # Verify request was made with correct params
            mock_request.assert_called_once_with(
                "POST",
                "/api/knowledge/similarity/batch",
                json={
                    "pairs": pairs,
                    "use_cache": True
                }
            )

            # Verify response parsing
            assert isinstance(result, BatchSimilarityResponse)
            assert result.count == 2
            assert len(result.results) == 2
            assert result.results[0].index == 0
            assert result.results[0].similarity == 0.95
            assert result.results[1].index == 1
            assert result.results[1].similarity == 0.60

    def test_compute_batch_similarity_empty_pairs(self, mock_client):
        """Test compute_batch_similarity with empty pairs."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "results": [],
                "model": "bge-m3-onnx",
                "count": 0,
                "execution_time_ms": 10.0
            }

            result = mock_client.compute_batch_similarity([])

            assert result.count == 0
            assert len(result.results) == 0

    def test_compute_batch_similarity_without_cache(self, mock_client):
        """Test compute_batch_similarity with cache disabled."""
        with patch.object(mock_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "results": [],
                "model": "bge-m3-onnx",
                "count": 0,
                "execution_time_ms": 10.0
            }

            mock_client.compute_batch_similarity(
                [{"text1": "a", "text2": "b"}],
                use_cache=False
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["use_cache"] is False


class TestSimilarityAPIRetry:
    """Test retry behavior for similarity API."""

    def test_retry_on_connection_error(self):
        """Test that connection errors trigger retries."""
        client = KnowledgeClient(base_url="http://mock-service:8000", max_retries=3)

        call_count = 0

        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise KnowledgeConnectionError("Connection failed")
            return {
                "similarity": 0.85,
                "model": "bge-m3-onnx",
                "execution_time_ms": 50.0
            }

        with patch.object(client, '_make_request', side_effect=mock_request):
            result = client.compute_similarity("text1", "text2")
            assert result.similarity == 0.85
            assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that exceeding max retries raises exception."""
        client = KnowledgeClient(base_url="http://mock-service:8000", max_retries=2)

        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = KnowledgeConnectionError("Connection failed")

            with pytest.raises(KnowledgeConnectionError):
                client.compute_similarity("text1", "text2")


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
class TestSimilarityAPIIntegration:
    """Integration tests for similarity API (requires running service)."""

    def test_health_check(self, live_client):
        """Test service health check."""
        health = live_client.health_check()
        assert health["status"] == "healthy"

    def test_identical_texts_similarity(self, live_client):
        """Test that identical texts have similarity 1.0."""
        text = "Login button should be visible on the page"
        result = live_client.compute_similarity(text, text)

        assert result.similarity == 1.0
        assert result.model == "bge-m3-onnx"
        assert result.execution_time_ms > 0

    def test_similar_texts_high_similarity(self, live_client):
        """Test that semantically similar texts have high similarity."""
        result = live_client.compute_similarity(
            text1="Login button should be visible",
            text2="The login button must be displayed"
        )

        # Similar texts should have >0.7 similarity
        assert result.similarity > 0.7
        assert result.similarity <= 1.0

    def test_different_texts_low_similarity(self, live_client):
        """Test that different texts have low similarity."""
        result = live_client.compute_similarity(
            text1="Login button should be visible",
            text2="The weather is nice today"
        )

        # Completely different texts should have <0.5 similarity
        assert result.similarity < 0.5

    def test_batch_similarity(self, live_client):
        """Test batch similarity computation."""
        pairs = [
            {"text1": "Login button visible", "text2": "Login button displayed"},
            {"text1": "Submit form", "text2": "Submit the form"},
            {"text1": "Email validation", "text2": "The weather is sunny"},
        ]
        result = live_client.compute_batch_similarity(pairs)

        assert result.count == 3
        assert len(result.results) == 3

        # First two pairs should have high similarity
        assert result.results[0].similarity > 0.7
        assert result.results[1].similarity > 0.7

        # Last pair should have low similarity
        assert result.results[2].similarity < 0.5

    def test_batch_similarity_shared_texts(self, live_client):
        """Test batch similarity with shared texts (should reuse embeddings)."""
        shared_text = "Common text used in multiple pairs"
        pairs = [
            {"text1": shared_text, "text2": "First comparison"},
            {"text1": shared_text, "text2": "Second comparison"},
            {"text1": shared_text, "text2": shared_text},  # identical
        ]
        result = live_client.compute_batch_similarity(pairs)

        assert result.count == 3
        # Last pair (identical) should be 1.0
        assert result.results[2].similarity == 1.0

    def test_unicode_text_similarity(self, live_client):
        """Test similarity for unicode text."""
        result = live_client.compute_similarity(
            text1="日本語のテキスト",  # Japanese
            text2="日本語のテキスト"
        )

        assert result.similarity == 1.0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run unit tests by default, integration tests if service available
    pytest.main([__file__, "-v", "--tb=short"])
