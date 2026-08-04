"""
SearchKnowledgeClient

Synchronous HTTP client for dxo-search-knowledge service.
Hybrid search (semantic + BM25 -> RRF -> rerank) over pre-chunked documents
(push-API pattern: chunking/entity extraction happens on the caller's side).

Usage:
    from dxo_client import SearchKnowledgeClient

    client = SearchKnowledgeClient(collection="collection-123")

    # Index pre-cut chunks (idempotent when chunk_id is omitted)
    result = client.ingest_chunks(
        doc_id="doc-001",
        chunks=[
            {"text": "Điều khoản bảo hành 12 tháng...", "metadata": {"section": "bảo hành"}},
            "Plain strings are accepted too",
        ],
        metadata={"file_name": "policy.pdf"},  # document-level, merged into every chunk
    )

    # Hybrid search
    response = client.search("thời hạn bảo hành", top_k=5)
    for hit in response.hits:
        print(hit.score, hit.text)
    print(response.combined_context)

    # Delete a document (all of its chunks)
    client.delete_document("doc-001")

Per-collection:
    # collection is sent as the X-Collection-Name header on every request.
    # Set it once on the client, or override per call:
    client.search("query", collection="other-collection")
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Callable, Union
from functools import wraps

import httpx

from .models.search_knowledge import (
    SearchChunk,
    IngestChunksResult,
    SearchHit,
    SearchQueryResult,
)

logger = logging.getLogger(__name__)


def _get_config(key: str, default: str) -> str:
    """
    Get config value with fallback hierarchy:
    1. Consul (if available)
    2. Environment variable
    3. Default value
    """
    try:
        from dxo_client import consul_client
        if consul_client is not None and consul_client.is_available():
            value = consul_client.get(key, None)
            if value is not None:
                return value
    except (ImportError, Exception):
        pass
    return os.getenv(key, default)


class SearchKnowledgeError(Exception):
    """Base exception for search knowledge client errors."""
    pass


class SearchKnowledgeConnectionError(SearchKnowledgeError):
    """Raised when connection to search knowledge service fails."""
    pass


class SearchKnowledgeTimeoutError(SearchKnowledgeError):
    """Raised when request to search knowledge service times out."""
    pass


class SearchKnowledgeAPIError(SearchKnowledgeError):
    """Raised when search knowledge service returns an error response."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class SearchKnowledgeNotFoundError(SearchKnowledgeError):
    """Raised when requested resource is not found."""
    pass


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.5):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except (SearchKnowledgeConnectionError, SearchKnowledgeTimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class SearchKnowledgeClient:
    """
    Synchronous HTTP client for dxo-search-knowledge service.

    Features:
    - Push-API chunk ingestion (client-side chunking)
    - Hybrid search: semantic + BM25 -> fusion -> rerank
    - Per-collection via X-Collection-Name header
    - Connection pooling
    - Retry with exponential backoff
    - Request interceptors
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        collection: Optional[str] = None,
        read_timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        interceptors: Optional[List[Callable[[Dict[str, str]], Dict[str, str]]]] = None
    ):
        """
        Initialize SearchKnowledgeClient.

        Args:
            base_url: Service URL (default: from SEARCH_KNOWLEDGE_SERVICE_URL env
                or http://localhost:8080)
            collection: Default collection, sent as the X-Collection-Name header on every
                request (default: from SEARCH_KNOWLEDGE_COLLECTION env, else None ->
                the service's default collection, if it allows collection-less requests)
            read_timeout: Read timeout in seconds (default: 30)
            connect_timeout: Connection timeout in seconds (default: 10)
            max_retries: Maximum retry attempts
            interceptors: List of request interceptors (applied after the
                X-Collection-Name header, so an interceptor can still override it)
        """
        self._base_url = base_url or _get_config("SEARCH_KNOWLEDGE_SERVICE_URL", "http://localhost:8080")
        self._collection = collection or _get_config("SEARCH_KNOWLEDGE_COLLECTION", "") or None
        self._timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._max_retries = max_retries
        self._client: Optional[httpx.Client] = None
        self._interceptors: List[Callable[[Dict[str, str]], Dict[str, str]]] = interceptors or []

    @property
    def base_url(self) -> str:
        """Get base URL."""
        return self._base_url

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout
            )
        return self._client

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        if response.status_code in (200, 201):
            return response.json()
        elif response.status_code == 404:
            raise SearchKnowledgeNotFoundError(f"Resource not found: {response.text}")
        elif response.status_code >= 500:
            raise SearchKnowledgeAPIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text
            )
        else:
            raise SearchKnowledgeAPIError(
                f"API error: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text
            )

    def _apply_interceptors(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply all registered interceptors to headers."""
        for interceptor in self._interceptors:
            headers = interceptor(headers)
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            client = self._get_client()

            headers = {"Content-Type": "application/json"}
            effective_collection = collection or self._collection
            if effective_collection:
                headers["X-Collection-Name"] = effective_collection
            headers = self._apply_interceptors(headers)

            response = client.request(
                method=method,
                url=endpoint,
                json=json,
                params=params,
                headers=headers
            )
            return self._handle_response(response)
        except httpx.ConnectError as e:
            raise SearchKnowledgeConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise SearchKnowledgeTimeoutError(f"Request timed out: {e}")

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # HEALTH / INTROSPECTION
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check search knowledge service health."""
        try:
            client = self._get_client()
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def readiness_check(self) -> Dict[str, Any]:
        """
        Check readiness of the service's dependencies (Qdrant, cache).

        Returns:
            {"qdrant": bool, "cache": bool|None, "status": "ready"|"degraded"}
        """
        return self._request("GET", "/health/ready")

    @retry_with_backoff(max_retries=3)
    def list_strategies(self) -> Dict[str, Any]:
        """
        List available/enabled retrievers and fusion strategies.

        Returns:
            {"retrievers": {"available": [...], "enabled": [...]},
             "fusion": {"available": [...], "active": str}}
        """
        return self._request("GET", "/v1/strategies")

    # =========================================================================
    # INGESTION (push-API: chunks are pre-cut by the caller)
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def ingest_chunks(
        self,
        doc_id: str,
        chunks: List[Union[str, Dict[str, Any], SearchChunk]],
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> IngestChunksResult:
        """
        Index pre-cut chunks for one document.

        Each chunk keeps its own metadata, merged over the document-level
        metadata. When chunk_id is omitted the service derives a stable ID
        from hash(doc_id + text), so re-ingesting the same content is
        idempotent. If provided, chunk_id must be a UUID string.

        Args:
            doc_id: Document identifier the chunks belong to
            chunks: List of chunks; each item is a plain string, a dict
                {"text": str, "chunk_id"?: str, "parent_id"?: str, "metadata"?: dict},
                or a SearchChunk
            metadata: Document-level metadata merged into every chunk
            collection: Override the client's default collection for this call

        Returns:
            IngestChunksResult(doc_id, chunks_indexed, chunk_ids)
        """
        normalized = []
        for chunk in chunks:
            if isinstance(chunk, str):
                chunk = SearchChunk(text=chunk)
            elif isinstance(chunk, dict):
                chunk = SearchChunk(**chunk)
            normalized.append(chunk.model_dump(exclude_none=True))

        data = self._request(
            "POST",
            "/v1/chunks",
            json={
                "doc_id": doc_id,
                "chunks": normalized,
                "metadata": metadata or {},
            },
            collection=collection,
        )
        return IngestChunksResult(**data)

    @retry_with_backoff(max_retries=3)
    def delete_document(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> bool:
        """
        Delete a document and all of its indexed chunks.

        Returns:
            True when the service acknowledged the deletion
        """
        result = self._request(
            "DELETE",
            f"/v1/documents/{doc_id}",
            collection=collection,
        )
        return bool(result.get("deleted", False))

    # =========================================================================
    # QUERY (hybrid search)
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrievers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> SearchQueryResult:
        """
        Hybrid search: semantic + BM25 -> fusion -> rerank.

        Args:
            query: Search query text
            top_k: Number of hits to return (1-100)
            retrievers: Subset of enabled retrievers to use
                (default: None -> all retrievers enabled in the service config)
            filters: Filter on chunk metadata, e.g. {"section": "bảo hành",
                "lang": ["vi", "en"]}. Key "doc_id" matches the document ID;
                other keys match inside metadata. A list value matches ANY.
            collection: Override the client's default collection for this call

        Returns:
            SearchQueryResult with ranked hits (use .combined_context for RAG)
        """
        body: Dict[str, Any] = {"query": query, "top_k": top_k}
        if retrievers is not None:
            body["retrievers"] = retrievers
        if filters is not None:
            body["filters"] = filters

        data = self._request("POST", "/v1/query", json=body, collection=collection)
        return SearchQueryResult(
            query=data.get("query", query),
            hits=[SearchHit(**hit) for hit in data.get("hits", [])],
        )

    def rag_retrieval(
        self,
        query: str,
        top_k: int = 10,
        retrievers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> str:
        """
        Convenience wrapper: run search() and return the combined context string.
        """
        return self.search(
            query,
            top_k=top_k,
            retrievers=retrievers,
            filters=filters,
            collection=collection,
        ).combined_context


# Singleton instance
_search_knowledge_client: Optional[SearchKnowledgeClient] = None


def get_search_knowledge_client() -> SearchKnowledgeClient:
    """Get singleton SearchKnowledgeClient instance."""
    global _search_knowledge_client
    if _search_knowledge_client is None:
        _search_knowledge_client = SearchKnowledgeClient()
    return _search_knowledge_client
