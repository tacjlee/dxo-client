"""
Test SearchKnowledgeClient (dxo-search-knowledge service)

Unit tests using pytest-httpx mocks — no running service required.

Run with: pytest tests/test_search_knowledge_client.py -v
"""

import json

import pytest

from dxo_client import (
    SearchKnowledgeClient,
    SearchChunk,
    IngestChunksResult,
    SearchQueryResult,
    SearchKnowledgeAPIError,
    SearchKnowledgeNotFoundError,
)

BASE_URL = "http://search-knowledge.test"


@pytest.fixture
def client():
    with SearchKnowledgeClient(base_url=BASE_URL, collection="collection-123") as c:
        yield c


class TestIngestChunks:
    def test_ingest_normalizes_str_dict_and_model_chunks(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/chunks",
            json={"doc_id": "doc-1", "chunks_indexed": 3, "chunk_ids": ["a", "b", "c"]},
        )

        result = client.ingest_chunks(
            doc_id="doc-1",
            chunks=[
                "plain string chunk",
                {"text": "dict chunk", "metadata": {"section": "warranty"}},
                SearchChunk(text="model chunk", parent_id="p-1"),
            ],
            metadata={"file_name": "policy.pdf"},
        )

        assert isinstance(result, IngestChunksResult)
        assert result.chunks_indexed == 3
        assert result.chunk_ids == ["a", "b", "c"]

        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["doc_id"] == "doc-1"
        assert body["metadata"] == {"file_name": "policy.pdf"}
        assert body["chunks"][0] == {"text": "plain string chunk", "metadata": {}}
        assert body["chunks"][1] == {"text": "dict chunk", "metadata": {"section": "warranty"}}
        assert body["chunks"][2] == {"text": "model chunk", "parent_id": "p-1", "metadata": {}}
        # chunk_id omitted -> excluded from payload so the service auto-generates it
        assert "chunk_id" not in body["chunks"][0]

    def test_ingest_sends_collection_header(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/chunks",
            json={"doc_id": "doc-1", "chunks_indexed": 1, "chunk_ids": ["x"]},
        )
        client.ingest_chunks(doc_id="doc-1", chunks=["hello"])
        assert httpx_mock.get_request().headers["X-Collection-Name"] == "collection-123"

    def test_per_call_collection_overrides_default(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/chunks",
            json={"doc_id": "doc-1", "chunks_indexed": 1, "chunk_ids": ["x"]},
        )
        client.ingest_chunks(doc_id="doc-1", chunks=["hello"], collection="other-collection")
        assert httpx_mock.get_request().headers["X-Collection-Name"] == "other-collection"


class TestSearch:
    def test_search_parses_hits(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/query",
            json={
                "query": "warranty period",
                "hits": [
                    {
                        "chunk_id": "c-1",
                        "doc_id": "doc-1",
                        "score": 0.92,
                        "text": "Warranty lasts 12 months.",
                        "source": "rrf",
                        "metadata": {"section": "warranty"},
                    },
                    {
                        "chunk_id": "c-2",
                        "doc_id": "doc-2",
                        "score": 0.71,
                        "text": "Extended warranty available.",
                        "source": "semantic",
                        "metadata": {},
                    },
                ],
            },
        )

        result = client.search("warranty period", top_k=5, filters={"lang": ["vi", "en"]})

        assert isinstance(result, SearchQueryResult)
        assert len(result.hits) == 2
        assert result.hits[0].chunk_id == "c-1"
        assert result.hits[0].score == pytest.approx(0.92)
        assert result.texts == ["Warranty lasts 12 months.", "Extended warranty available."]
        assert "12 months" in result.combined_context

        body = json.loads(httpx_mock.get_request().content)
        assert body == {
            "query": "warranty period",
            "top_k": 5,
            "filters": {"lang": ["vi", "en"]},
        }

    def test_search_omits_optional_fields(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/query",
            json={"query": "q", "hits": []},
        )
        client.search("q")
        body = json.loads(httpx_mock.get_request().content)
        assert body == {"query": "q", "top_k": 10}

    def test_rag_retrieval_returns_combined_context(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/query",
            json={
                "query": "q",
                "hits": [
                    {"chunk_id": "c-1", "doc_id": "d", "score": 1.0, "text": "first", "source": "rrf"},
                    {"chunk_id": "c-2", "doc_id": "d", "score": 0.5, "text": "second", "source": "rrf"},
                ],
            },
        )
        assert client.rag_retrieval("q") == "first\n\nsecond"

    def test_bad_request_raises_api_error(self, client, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/query",
            status_code=400,
            json={"detail": "Retriever không khả dụng: ['nope']"},
        )
        with pytest.raises(SearchKnowledgeAPIError) as exc_info:
            client.search("q", retrievers=["nope"])
        assert exc_info.value.status_code == 400


class TestDeleteDocument:
    def test_delete_document(self, client, httpx_mock):
        httpx_mock.add_response(
            method="DELETE",
            url=f"{BASE_URL}/v1/documents/doc-1",
            json={"doc_id": "doc-1", "deleted": True},
        )
        assert client.delete_document("doc-1") is True

    def test_delete_missing_document_raises_not_found(self, client, httpx_mock):
        httpx_mock.add_response(
            method="DELETE",
            url=f"{BASE_URL}/v1/documents/ghost",
            status_code=404,
            json={"detail": "not found"},
        )
        with pytest.raises(SearchKnowledgeNotFoundError):
            client.delete_document("ghost")


class TestHealthAndStrategies:
    def test_health_check(self, client, httpx_mock):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/health",
            json={"service": "dxo-search-knowledge", "version": "0.1.0", "status": "ok"},
        )
        assert client.health_check()["status"] == "ok"

    def test_health_check_swallows_errors(self):
        client = SearchKnowledgeClient(base_url="http://localhost:1")  # nothing listens here
        assert client.health_check()["status"] == "unhealthy"

    def test_list_strategies(self, client, httpx_mock):
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/v1/strategies",
            json={
                "retrievers": {"available": ["semantic", "bm25"], "enabled": ["semantic", "bm25"]},
                "fusion": {"available": ["rrf"], "active": "rrf"},
            },
        )
        strategies = client.list_strategies()
        assert "bm25" in strategies["retrievers"]["enabled"]


class TestInterceptors:
    def test_interceptor_can_add_and_override_headers(self, httpx_mock):
        def auth(headers):
            headers["Authorization"] = "Bearer token-1"
            headers["X-Collection-Name"] = "intercepted"
            return headers

        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/v1/query",
            json={"query": "q", "hits": []},
        )
        with SearchKnowledgeClient(base_url=BASE_URL, collection="collection-123", interceptors=[auth]) as client:
            client.search("q")

        request = httpx_mock.get_request()
        assert request.headers["Authorization"] == "Bearer token-1"
        assert request.headers["X-Collection-Name"] == "intercepted"
