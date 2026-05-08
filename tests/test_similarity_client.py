"""
Test KnowledgeClient Similarity Methods

Run with: pytest tests/test_similarity_client.py -v
Or standalone: python tests/test_similarity_client.py

Requires workflow-knowledge service running on localhost:8001
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxo_client import KnowledgeClient
from dxo_client.models import SearchRecordsResponse, RecordMatch


class TestGenerateMultivectorEmbeddings:
    """Test KnowledgeClient.generate_multivector_embeddings()"""

    @pytest.fixture
    def client(self):
        return KnowledgeClient(base_url="http://localhost:8001")

    def test_generate_embeddings_returns_plain_dicts(self, client):
        """Test that embeddings are returned as plain dicts for PostgreSQL."""
        embeddings = client.generate_multivector_embeddings(
            texts=["検索ボタン", "Submit button"],
            include_sparse=True,
            include_colbert=False,
        )

        # Should return list of plain dicts
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

        # Each embedding should be a plain dict
        for emb in embeddings:
            assert isinstance(emb, dict)
            assert "dense" in emb
            assert isinstance(emb["dense"], list)
            assert len(emb["dense"]) == 1024  # BGE-M3 dimension

            # Sparse should be present when requested
            assert "sparse" in emb
            assert isinstance(emb["sparse"], dict)

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Dense dimension: {len(embeddings[0]['dense'])}")
        print(f"Sparse tokens: {len(embeddings[0]['sparse'])}")

    def test_embeddings_can_be_json_serialized(self, client):
        """Test that embeddings can be serialized to JSON (for PostgreSQL JSONB)."""
        import json

        embeddings = client.generate_multivector_embeddings(
            texts=["Test text"],
            include_sparse=True,
        )

        # Should be JSON serializable
        json_str = json.dumps(embeddings[0])
        assert isinstance(json_str, str)

        # Should be deserializable
        restored = json.loads(json_str)
        assert restored["dense"] == embeddings[0]["dense"]

        print(f"JSON size: {len(json_str)} bytes")

    def test_embeddings_without_sparse(self, client):
        """Test embedding generation without sparse vectors."""
        embeddings = client.generate_multivector_embeddings(
            texts=["Test text"],
            include_sparse=False,
        )

        emb = embeddings[0]
        assert "dense" in emb
        # sparse should be None or not present
        assert emb.get("sparse") is None


class TestSearchRecords:
    """Test KnowledgeClient.search_records()"""

    @pytest.fixture
    def client(self):
        return KnowledgeClient(base_url="http://localhost:8001")

    def test_search_with_text_records(self, client):
        """Test search with text records."""
        result = client.search_records(
            query="検索",
            records=[
                {"id": "1", "text": "検索ボタン Search button"},
                {"id": "2", "text": "Submit form button"},
                {"id": "3", "text": "Cancel operation"},
            ],
            top_k=3,
        )

        # Check return type
        assert isinstance(result, SearchRecordsResponse)
        assert result.total_records == 3
        assert result.texts_embedded == 4  # 3 records + 1 query

        # Check matches
        for match in result.matches:
            assert isinstance(match, RecordMatch)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
            assert 0.0 <= match.score <= 1.0

        print(f"Total records: {result.total_records}")
        print(f"Texts embedded: {result.texts_embedded}")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")
        for match in result.matches:
            print(f"  {match.id}: {match.score:.4f}")

    def test_search_with_precomputed_embeddings(self, client):
        """Test search with pre-computed embeddings (PostgreSQL use case)."""
        # Step 1: Generate embeddings (simulating PostgreSQL storage)
        texts = ["検索ボタン", "Submit button", "Cancel"]
        embeddings = client.generate_multivector_embeddings(texts)

        # Step 2: Create records with embeddings (simulating query from PostgreSQL)
        records = [
            {"id": "intent-1", "embedding": embeddings[0], "metadata": {"name": "Search"}},
            {"id": "intent-2", "embedding": embeddings[1], "metadata": {"name": "Submit"}},
            {"id": "intent-3", "embedding": embeddings[2], "metadata": {"name": "Cancel"}},
        ]

        # Step 3: Search
        result = client.search_records(
            query="検索",
            records=records,
            top_k=3,
        )

        # Only query should be embedded (records have pre-computed)
        assert result.texts_embedded == 1
        assert result.total_records == 3

        # First match should be search button
        assert result.matches[0].id == "intent-1"

        print(f"Pre-computed search - texts_embedded: {result.texts_embedded}")
        for match in result.matches:
            print(f"  {match.id}: {match.score:.4f} - {match.metadata}")

    def test_search_with_min_similarity(self, client):
        """Test min_similarity filtering."""
        result = client.search_records(
            query="検索",
            records=[
                {"id": "1", "text": "検索ボタン"},
                {"id": "2", "text": "Weather forecast today"},
            ],
            min_similarity=0.5,
        )

        # All matches should have score >= 0.5
        for match in result.matches:
            assert match.score >= 0.5

    def test_search_with_custom_weights(self, client):
        """Test custom similarity weights."""
        result = client.search_records(
            query="検索ボタン",
            records=[
                {"id": "1", "text": "検索ボタン Search"},
                {"id": "2", "text": "Search button click"},
            ],
            weights={"dense": 0.7, "sparse": 0.3},
        )

        assert len(result.matches) > 0
        print("Custom weights (dense=0.7, sparse=0.3):")
        for match in result.matches:
            print(f"  {match.id}: {match.score:.4f}")


def run_tests():
    """Run tests manually without pytest."""
    print("=" * 60)
    print("Testing KnowledgeClient Similarity Methods")
    print("=" * 60)

    client = KnowledgeClient(base_url="http://localhost:8001")

    # Check service health
    health = client.health_check()
    if health["status"] != "healthy":
        print(f"\nERROR: Service not healthy: {health}")
        print("Start the service first: cd workflow-knowledge-repo && python -m uvicorn app.main:app --port 8001")
        return

    print(f"\nService health: {health['status']}")

    print("\n" + "-" * 60)
    print("Test 1: Generate Multivector Embeddings")
    print("-" * 60)
    try:
        embeddings = client.generate_multivector_embeddings(
            texts=["検索ボタン", "Submit button"],
            include_sparse=True,
        )
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert isinstance(embeddings[0], dict)
        assert "dense" in embeddings[0]
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Dense dimension: {len(embeddings[0]['dense'])}")
        print(f"Sparse tokens: {len(embeddings[0].get('sparse', {}))}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "-" * 60)
    print("Test 2: Search Records with Text")
    print("-" * 60)
    try:
        result = client.search_records(
            query="検索",
            records=[
                {"id": "1", "text": "検索ボタン Search button"},
                {"id": "2", "text": "Submit form button"},
                {"id": "3", "text": "Cancel operation"},
            ],
            top_k=3,
        )
        assert isinstance(result, SearchRecordsResponse)
        assert result.total_records == 3
        print(f"Total: {result.total_records}, Embedded: {result.texts_embedded}")
        for match in result.matches:
            print(f"  {match.id}: {match.score:.4f}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "-" * 60)
    print("Test 3: Search with Pre-computed Embeddings")
    print("-" * 60)
    try:
        # Generate embeddings
        embeddings = client.generate_multivector_embeddings(
            texts=["検索ボタン", "Submit", "Cancel"]
        )

        # Search with pre-computed
        result = client.search_records(
            query="検索",
            records=[
                {"id": "1", "embedding": embeddings[0]},
                {"id": "2", "embedding": embeddings[1]},
                {"id": "3", "embedding": embeddings[2]},
            ],
        )
        assert result.texts_embedded == 1  # Only query embedded
        print(f"Texts embedded: {result.texts_embedded} (only query)")
        for match in result.matches:
            print(f"  {match.id}: {match.score:.4f}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    run_tests()
