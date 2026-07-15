"""
KnowledgeClient

Synchronous HTTP client for workflow-knowledge service.
Similar to Java FeignClient pattern with service discovery and retry logic.

Usage:
    from dxo_client import KnowledgeClient, MetadataFilter

    client = KnowledgeClient()

    # Create collection (tenant-scoped)
    client.create_collection("tenant-123", "my-collection")
    # Creates: tenant_tenant_123_my_collection

    # Add documents (with full hierarchy)
    result = client.add_documents(
        collection_name="tenant_tenant_123_my_collection",
        documents=[{"content": "Hello world", "metadata": {"file_name": "doc.pdf"}}],
        tenant_id="tenant-123",
        knowledge_id="kb-789"
    )

    # Search with tenant filtering
    results = client.search(
        collection_name="tenant_tenant_123_my_collection",
        query="hello",
        top_k=10,
        filters=MetadataFilter(tenant_id="tenant-123", knowledge_id="kb-789")
    )
"""

import time
import logging
from typing import Optional, List, Dict, Any, Callable, Protocol
from functools import wraps

import httpx


class RequestInterceptor(Protocol):
    """
    Request interceptor protocol (similar to Java FeignClient RequestInterceptor).

    Implement this protocol to add headers, modify requests, or add tracing.

    Example:
        class AuthInterceptor:
            def __init__(self, token: str):
                self.token = token

            def __call__(self, headers: Dict[str, str]) -> Dict[str, str]:
                headers["Authorization"] = f"Bearer {self.token}"
                return headers

        client = KnowledgeClient(interceptors=[AuthInterceptor("my-token")])
    """
    def __call__(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Intercept and modify request headers.

        Args:
            headers: Current request headers

        Returns:
            Modified headers dict
        """
        ...


from .models import (
    MetadataFilter,
    CollectionInfo,
    SearchResult,
    RAGContext,
    DocumentChunk,
    DocumentProcessResult,
    ExtractionResult,
    SupportedFormats,
    ParentChildChunkConfig,
    ParentChildProcessResult,
    ParentResult,
    SearchExpandResult,
    SimilarityResponse,
    BatchSimilarityItem,
    BatchSimilarityResult,
    BatchSimilarityResponse,
    RecordMatch,
    SearchRecordsResponse,
)
from .exceptions import (
    KnowledgeConnectionError,
    KnowledgeTimeoutError,
    KnowledgeAPIError,
    KnowledgeNotFoundError,
    KnowledgeValidationError,
    KnowledgeCircuitBreakerError,
)
from .service_discovery import ServiceDiscovery

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.5):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except (KnowledgeConnectionError, KnowledgeTimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class KnowledgeClient:
    """
    Synchronous HTTP client for workflow-knowledge service.

    Features:
    - Service discovery via Consul with environment fallback
    - Connection pooling
    - Retry with exponential backoff
    - Type-safe request/response models
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        read_timeout: float = 600.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        max_connections_per_route: int = 50,
        max_connections: int = 200,
        interceptors: Optional[List[Callable[[Dict[str, str]], Dict[str, str]]]] = None
    ):
        """
        Initialize KnowledgeClient.

        Args:
            base_url: Direct URL override (bypasses service discovery)
            read_timeout: Read timeout in seconds (default: 600, increased for large document processing)
            connect_timeout: Connection timeout in seconds (default: 10, same as Java FeignClient)
            max_retries: Maximum retry attempts
            max_connections_per_route: Max keepalive connections per route (default: 50, same as Java FeignClient)
            max_connections: Max total connections (default: 200, same as Java FeignClient)
            interceptors: List of request interceptors (similar to Java FeignClient RequestInterceptor)
        """
        self._service_discovery = ServiceDiscovery()
        self._base_url = base_url
        self._timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._max_retries = max_retries
        self._limits = httpx.Limits(
            max_keepalive_connections=max_connections_per_route,
            max_connections=max_connections
        )
        self._client: Optional[httpx.Client] = None
        self._interceptors: List[Callable[[Dict[str, str]], Dict[str, str]]] = interceptors or []

    @property
    def base_url(self) -> str:
        """Get base URL from service discovery or fallback."""
        if self._base_url:
            return self._base_url
        return self._service_discovery.get_knowledge_service_url()

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self._timeout,
                limits=self._limits
            )
        return self._client

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        if response.status_code in (200, 201):
            return response.json()
        elif response.status_code == 404:
            raise KnowledgeNotFoundError(f"Resource not found: {response.text}")
        elif response.status_code == 422:
            raise KnowledgeValidationError(f"Validation error: {response.text}")
        elif response.status_code == 503:
            raise KnowledgeCircuitBreakerError(f"Service unavailable (circuit breaker open): {response.text}")
        elif response.status_code >= 500:
            raise KnowledgeAPIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text
            )
        else:
            raise KnowledgeAPIError(
                f"API error: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text
            )

    def _apply_interceptors(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply all registered interceptors to headers."""
        for interceptor in self._interceptors:
            headers = interceptor(headers)
        return headers

    def add_interceptor(self, interceptor: Callable[[Dict[str, str]], Dict[str, str]]) -> None:
        """
        Add a request interceptor.

        Args:
            interceptor: Callable that takes headers dict and returns modified headers

        Example:
            client.add_interceptor(lambda h: {**h, "X-Trace-Id": "abc123"})
        """
        self._interceptors.append(interceptor)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            client = self._get_client()

            # Apply interceptors to headers
            headers = {"Content-Type": "application/json"}
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
            raise KnowledgeConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise KnowledgeTimeoutError(f"Request timed out: {e}")

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # COLLECTION OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def create_collection(
        self,
        tenant_id: str,
        name: str,
        enable_multivector: bool = True,
        vector_size: int = 1024
    ) -> CollectionInfo:
        """
        Create a new vector collection.

        Args:
            tenant_id: Tenant ID for multi-tenant isolation
            name: Collection name suffix
            enable_multivector: Enable BGE-M3 multi-vector search (dense + sparse + ColBERT)
            vector_size: Vector dimension size (1024 for BGE-M3)

        Returns:
            CollectionInfo with created collection details
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/collections",
            json={
                "tenant_id": tenant_id,
                "name": name,
                "enable_multivector": enable_multivector,
                "vector_size": vector_size,
                "distance": "Cosine"
            }
        )
        return CollectionInfo(**data)

    @retry_with_backoff(max_retries=3)
    def create_collection_direct(
        self,
        collection_name: str,
        enable_multivector: bool = True,
        vector_size: int = 1024
    ) -> CollectionInfo:
        """
        Create a new vector collection with exact name.

        Use this for global collections (e.g., 'golden_data') that don't need tenant prefix.

        Args:
            collection_name: Exact collection name (no prefix added)
            enable_multivector: Enable BGE-M3 multi-vector search
            vector_size: Vector dimension size (1024 for BGE-M3)

        Returns:
            CollectionInfo with created collection details
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/collections",
            json={
                "collection_name": collection_name,
                "enable_multivector": enable_multivector,
                "vector_size": vector_size,
                "distance": "Cosine"
            }
        )
        return CollectionInfo(**data)

    @retry_with_backoff(max_retries=3)
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """Get collection information."""
        data = self._make_request("GET", f"/api/knowledge/collections/{collection_name}")
        return CollectionInfo(**data)

    @retry_with_backoff(max_retries=3)
    def delete_collection(
        self,
        collection_name: str,
        tenant_id: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Delete a collection."""
        params = {"force": force}
        if tenant_id:
            params["tenant_id"] = tenant_id

        self._make_request("DELETE", f"/api/knowledge/collections/{collection_name}", params=params)
        return True

    @retry_with_backoff(max_retries=3)
    def list_collections(self, tenant_id: Optional[str] = None) -> List[CollectionInfo]:
        """List collections for a tenant."""
        params = {}
        if tenant_id:
            params["tenant_id"] = tenant_id

        data = self._make_request("GET", "/api/knowledge/collections", params=params)
        return [CollectionInfo(**c) for c in data.get("collections", [])]

    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        tenant_id: str,
        knowledge_id: str,
        user_id: Optional[str] = None,
        document_type: str = "document",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> DocumentProcessResult:
        """
        Add documents to collection (with automatic chunking and embedding).

        Hierarchy: tenant_id -> knowledge_id -> document_id

        Args:
            collection_name: Target collection name
            documents: List of document dicts with 'content' and optional 'metadata'
            tenant_id: Tenant ID (required)
            knowledge_id: Knowledge base ID (required)
            user_id: Optional user ID
            document_type: Document type (e.g., document, template, viewpoint, rule)
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks

        Returns:
            DocumentProcessResult with chunking results
        """
        # For now, process each document individually
        # A batch endpoint could be added to the service
        all_chunks = []
        all_vector_ids = []

        for doc in documents:
            content = doc.get("content", "")
            file_name = doc.get("metadata", {}).get("file_name")
            document_id = doc.get("metadata", {}).get("document_id")
            # Allow per-document type override from metadata
            doc_type = doc.get("metadata", {}).get("document_type", document_type)

            data = self._make_request(
                "POST",
                "/api/knowledge/documents/process",
                json={
                    "collection_name": collection_name,
                    "content": content,
                    "tenant_id": tenant_id,
                    "knowledge_id": knowledge_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "user_id": user_id,
                    "document_type": doc_type,
                    "chunk_config": {
                        "strategy": "sentence",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap
                    }
                }
            )

            all_chunks.extend([DocumentChunk(**c) for c in data.get("chunks", [])])
            if data.get("vector_ids"):
                all_vector_ids.extend(data["vector_ids"])

        return DocumentProcessResult(
            document_id=documents[0].get("metadata", {}).get("document_id", "batch"),
            chunks_count=len(all_chunks),
            chunks=all_chunks,
            vector_ids=all_vector_ids if all_vector_ids else None,
            status="processed"
        )

    @retry_with_backoff(max_retries=3)
    def delete_documents(
        self,
        collection_name: str,
        tenant_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> int:
        """
        Delete document vectors from collection.

        At least one filter is required.
        Hierarchy: tenant_id -> knowledge_id -> document_id
        """
        data = self._make_request(
            "DELETE",
            "/api/knowledge/documents",
            json={
                "collection_name": collection_name,
                "tenant_id": tenant_id,
                "knowledge_id": knowledge_id,
                "document_id": document_id,
                "document_type": document_type,
                "file_name": file_name
            }
        )
        return data.get("deleted_count", 0)

    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]],
        auto_embed: bool = True
    ) -> List[str]:
        """
        Add vectors to collection.

        Args:
            collection_name: Collection name
            vectors: List of vector dicts with 'content', optional 'embedding', 'metadata'
            auto_embed: Generate embeddings if not provided

        Returns:
            List of added vector IDs
        """
        # Transform to API format
        api_vectors = []
        for v in vectors:
            vec = {
                "content": v["content"],
                "metadata": v.get("metadata", {})
            }
            if v.get("id"):
                vec["id"] = v["id"]
            if v.get("embedding"):
                vec["embedding"] = v["embedding"]
            api_vectors.append(vec)

        data = self._make_request(
            "POST",
            "/api/knowledge/vectors",
            json={
                "collection_name": collection_name,
                "vectors": api_vectors,
                "auto_embed": auto_embed
            }
        )
        return data.get("vector_ids", [])

    @retry_with_backoff(max_retries=3)
    def delete_vectors(
        self,
        collection_name: str,
        vector_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete vectors by IDs or filter."""
        data = self._make_request(
            "DELETE",
            "/api/knowledge/vectors",
            json={
                "collection_name": collection_name,
                "vector_ids": vector_ids,
                "filters": filters
            }
        )
        return data.get("deleted_count", 0)

    # =========================================================================
    # EMBEDDING OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        data = self._make_request(
            "POST",
            "/api/knowledge/embeddings",
            json={
                "texts": texts,
                "batch_size": batch_size,
                "use_cache": True
            }
        )
        return data.get("embeddings", [])

    # =========================================================================
    # SIMILARITY OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        use_cache: bool = True
    ) -> SimilarityResponse:
        """
        Compute semantic similarity between two texts.

        Uses BGE-M3 embeddings and cosine similarity.
        Useful for comparing test cases, documents, or any text pairs.

        Args:
            text1: First text (e.g., golden test case)
            text2: Second text (e.g., generated test case)
            use_cache: Use cache for embeddings (default: True)

        Returns:
            SimilarityResponse with similarity score (0.0-1.0)

        Example:
            result = client.compute_similarity(
                text1="Login button should be visible",
                text2="The login button must be displayed"
            )
            print(f"Similarity: {result.similarity}")  # e.g., 0.92
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/similarity",
            json={
                "text1": text1,
                "text2": text2,
                "use_cache": use_cache
            }
        )
        return SimilarityResponse(**data)

    @retry_with_backoff(max_retries=3)
    def compute_batch_similarity(
        self,
        pairs: List[Dict[str, str]],
        use_cache: bool = True
    ) -> BatchSimilarityResponse:
        """
        Compute semantic similarity for multiple text pairs.

        Optimized for batch processing - embeddings are generated in one batch
        and reused for pairs that share the same text.

        Args:
            pairs: List of dicts with 'text1' and 'text2' keys
            use_cache: Use cache for embeddings (default: True)

        Returns:
            BatchSimilarityResponse with similarity scores for each pair

        Example:
            result = client.compute_batch_similarity([
                {"text1": "Login button visible", "text2": "Login button displayed"},
                {"text1": "Submit form", "text2": "Click submit button"}
            ])
            for r in result.results:
                print(f"Pair {r.index}: {r.similarity}")
        """
        # Convert dicts to API format
        api_pairs = [
            {"text1": p["text1"], "text2": p["text2"]}
            for p in pairs
        ]

        data = self._make_request(
            "POST",
            "/api/knowledge/similarity/batch",
            json={
                "pairs": api_pairs,
                "use_cache": use_cache
            }
        )

        results = [BatchSimilarityResult(**r) for r in data.get("results", [])]
        return BatchSimilarityResponse(
            results=results,
            model=data.get("model", "bge-m3-onnx"),
            count=data.get("count", len(results)),
            execution_time_ms=data.get("execution_time_ms", 0.0)
        )

    @retry_with_backoff(max_retries=3)
    def generate_multivector_embeddings(
        self,
        texts: List[str],
        include_sparse: bool = True,
        include_colbert: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate multi-vector embeddings for texts.

        Returns plain dicts that can be directly stored in PostgreSQL JSONB columns.
        Each embedding contains dense vector (1024-dim) and optionally sparse vector.

        Args:
            texts: List of texts to embed
            include_sparse: Include sparse vectors (default: True)
            include_colbert: Include ColBERT vectors (default: False, large size)

        Returns:
            List of embedding dicts: [{"dense": [...], "sparse": {...}}, ...]

        Example:
            # Generate embeddings for storage
            embeddings = client.generate_multivector_embeddings(["検索ボタン", "Submit"])

            # Store in PostgreSQL JSONB column
            for text, emb in zip(texts, embeddings):
                intent.embedding = emb  # Direct assignment to JSONB column
                db.session.commit()
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/similarity/generate-embeddings",
            json={
                "texts": texts,
                "include_sparse": include_sparse,
                "include_colbert": include_colbert,
            }
        )

        # Return plain dicts for PostgreSQL storage
        return [emb for emb in data.get("embeddings", [])]

    @retry_with_backoff(max_retries=3)
    def search_records(
        self,
        query: str,
        records: List[Dict[str, Any]],
        top_k: int = 10,
        min_similarity: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> SearchRecordsResponse:
        """
        Search records using multi-vector similarity.

        Stateless semantic search - no data is stored. Records can provide
        either text (embedded on-the-fly) or pre-computed embeddings.

        Args:
            query: Query text (always text, never embedding)
            records: List of records, each with:
                - id: Record identifier
                - text: Text to embed (optional if embedding provided)
                - embedding: Pre-computed embedding dict (optional if text provided)
                - metadata: Optional metadata to return with matches
            top_k: Number of top matches to return (default: 10)
            min_similarity: Minimum similarity threshold (default: 0.0)
            weights: Similarity weights {"dense": 0.5, "sparse": 0.5}

        Returns:
            SearchRecordsResponse with matching records sorted by score

        Example with pre-computed embeddings (fast):
            records = [
                {"id": "1", "embedding": intent1.embedding, "metadata": {"name": "Search"}},
                {"id": "2", "embedding": intent2.embedding, "metadata": {"name": "Submit"}},
            ]
            result = client.search_records("検索ボタン", records, top_k=5)
            for match in result.matches:
                print(f"{match.id}: {match.score}")

        Example with text (slower, embeds on-the-fly):
            records = [
                {"id": "1", "text": "Search button click"},
                {"id": "2", "text": "Submit form action"},
            ]
            result = client.search_records("検索", records)
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/similarity/search-records",
            json={
                "query": query,
                "records": records,
                "top_k": top_k,
                "min_similarity": min_similarity,
                "weights": weights,
            }
        )

        matches = [RecordMatch(**m) for m in data.get("matches", [])]
        return SearchRecordsResponse(
            matches=matches,
            total_records=data.get("total_records", 0),
            texts_embedded=data.get("texts_embedded", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filters: Optional[MetadataFilter] = None,
        score_threshold: Optional[float] = None,
        include_embeddings: bool = False,
        search_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents using semantic similarity.

        Args:
            collection_name: Collection to search
            query: Query text
            top_k: Number of results
            filters: Metadata filters
            score_threshold: Minimum similarity score (0-1)
            include_embeddings: Include embeddings in results
            search_type: Search mode - "auto" (default routing), "bm25"
                (pure lexical), or "hybrid_rrf" (dense + sparse + bm25
                fused with RRF). Requires dxo-knowledge with BM25 support;
                omit for legacy behavior.

        Returns:
            List of SearchResult
        """
        request_data = {
            "collection_name": collection_name,
            "query": query,
            "top_k": top_k,
            "include_embeddings": include_embeddings
        }
        if filters:
            request_data["filters"] = filters.to_dict() if hasattr(filters, 'to_dict') else filters
        if score_threshold is not None:
            request_data["score_threshold"] = score_threshold
        if search_type is not None:
            request_data["search_type"] = search_type

        data = self._make_request("POST", "/api/knowledge/search", json=request_data)
        return [SearchResult(**r) for r in data.get("results", [])]

    @retry_with_backoff(max_retries=3)
    def search_by_document_type(
        self,
        collection_name: str,
        query: str,
        document_type: str,
        tenant_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        include_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search within a specific document type.

        Convenience method for filtering search results by document_type.
        Useful for searching specific categories like 'viewpoint', 'rule',
        'template', etc.

        Args:
            collection_name: Collection to search
            query: Query text
            document_type: Document type to filter by (e.g., 'viewpoint', 'rule')
            tenant_id: Optional tenant ID filter
            knowledge_id: Optional knowledge base ID filter
            top_k: Number of results (default: 10)
            score_threshold: Minimum similarity score (0-1)
            include_embeddings: Include embeddings in results

        Returns:
            List of SearchResult matching the document type

        Example:
            # Search for viewpoints about email validation
            results = client.search_by_document_type(
                collection_name="tenant_abc_test_kb",
                query="email validation",
                document_type="viewpoint",
                tenant_id="abc",
                top_k=5
            )
        """
        filters = MetadataFilter(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            document_type=document_type
        )
        return self.search(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
            include_embeddings=include_embeddings
        )

    @retry_with_backoff(max_retries=3)
    def rag_retrieval(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        filters: Optional[MetadataFilter] = None,
        rerank: bool = False,
        rerank_top_n: int = 3,
        search_type: Optional[str] = None
    ) -> RAGContext:
        """
        RAG retrieval: get relevant chunks for a query.

        Note: External reranker is disabled. For multivector collections,
        ColBERT late-interaction provides built-in reranking via hybrid_search.
        The rerank/rerank_top_n parameters are kept for API compatibility but ignored.

        Args:
            collection_name: Collection to search
            query: Query text
            top_k: Number of chunks to retrieve
            filters: Metadata filters
            rerank: Deprecated - ignored (ColBERT reranking is automatic)
            rerank_top_n: Deprecated - ignored
            search_type: Search mode - "auto", "bm25", or "hybrid_rrf".
                Requires dxo-knowledge with BM25 support; omit for legacy
                behavior.

        Returns:
            RAGContext with chunks and combined context
        """
        request_data = {
            "collection_name": collection_name,
            "query": query,
            "top_k": top_k,
            "rerank": rerank,
            "rerank_top_n": rerank_top_n
        }
        if filters:
            request_data["filters"] = filters.to_dict() if hasattr(filters, 'to_dict') else filters
        if search_type is not None:
            request_data["search_type"] = search_type

        data = self._make_request("POST", "/api/knowledge/search/rag", json=request_data)

        context_data = data.get("context", {})
        return RAGContext(
            chunks=[SearchResult(**c) for c in context_data.get("chunks", [])],
            combined_context=context_data.get("combined_context", ""),
            source_documents=context_data.get("source_documents", [])
        )

    # =========================================================================
    # PARENT-CHILD CHUNKING OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def add_documents_parent_child(
        self,
        collection_name: str,
        content: str,
        tenant_id: str,
        knowledge_id: str,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        user_id: Optional[str] = None,
        document_type: str = "document",
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        child_chunk_overlap: Optional[int] = None
    ) -> ParentChildProcessResult:
        """
        Add document with parent-child chunking strategy.

        Creates large parent chunks (for content retrieval) and small child chunks
        (for semantic search). Search returns full parent content for completeness.

        Parent chunks use placeholder embeddings (not searchable) to avoid ColBERT truncation.
        Only child chunks have real embeddings and are searchable.

        Ideal for test cases, viewpoints, decision tables, and other content
        where completeness matters.

        Args:
            collection_name: Target collection name
            content: Full document content
            tenant_id: Tenant ID (required)
            knowledge_id: Knowledge base ID (required)
            document_id: Document ID (auto-generated if not provided)
            file_name: Original file name
            user_id: Optional user ID
            document_type: Document type (document, viewpoint, testcase, etc.)
            parent_chunk_size: Size of parent chunks (default 10000 chars)
            child_chunk_size: Size of child chunks (default 1000 chars)
            child_chunk_overlap: Overlap between children (default 200)

        Returns:
            ParentChildProcessResult with parent and child IDs

        Example:
            result = client.add_documents_parent_child(
                collection_name="tenant_abc_test_kb",
                content=test_case_content,
                tenant_id="abc",
                knowledge_id="kb-123",
                document_type="testcase",
                parent_chunk_size=2000,
                child_chunk_size=200
            )
            print(f"Created {result.parent_count} parents, {result.child_count} children")
        """
        # Build chunk_config only with provided values (let server use defaults)
        chunk_config = {}
        if parent_chunk_size is not None:
            chunk_config["parent_chunk_size"] = parent_chunk_size
        if child_chunk_size is not None:
            chunk_config["child_chunk_size"] = child_chunk_size
        if child_chunk_overlap is not None:
            chunk_config["child_chunk_overlap"] = child_chunk_overlap

        request_body = {
            "collection_name": collection_name,
            "content": content,
            "tenant_id": tenant_id,
            "knowledge_id": knowledge_id,
            "document_id": document_id,
            "file_name": file_name,
            "user_id": user_id,
            "document_type": document_type,
        }
        if chunk_config:
            request_body["chunk_config"] = chunk_config

        data = self._make_request(
            "POST",
            "/api/knowledge/documents/process/parent-child",
            json=request_body
        )
        return ParentChildProcessResult(**data)

    @retry_with_backoff(max_retries=3)
    def search_for_parent(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        expand_top_n: int = 5,
        filters: Optional[MetadataFilter] = None,
        score_threshold: Optional[float] = None,
        include_children: bool = False,
        score_aggregation: str = "mean",
        search_type: Optional[str] = None
    ) -> SearchExpandResult:
        """
        Search for parent documents by querying child chunks.

        Ideal for retrieving complete test cases, viewpoints, or decision tables.

        This method:
        1. Searches child chunks (chunk_type="child")
        2. Groups results by parent_id
        3. Calculates parent score using aggregation method
        4. Returns full parent content for top N parents

        Args:
            collection_name: Collection to search
            query: Query text
            top_k: Number of children to search
            expand_top_n: Number of unique parents to return
            filters: Metadata filters (applied to children)
            score_threshold: Minimum similarity score
            include_children: Include matching children in response
            score_aggregation: How to aggregate child scores for parent ranking.
                - "mean": Average of matching child scores (default, better relevance)
                - "max": Maximum child score (legacy behavior)
                - "sum": Sum of matching child scores (favors more matches)
            search_type: Search mode - "auto", "bm25", or "hybrid_rrf".
                Requires dxo-knowledge with BM25 support; omit for legacy
                behavior.

        Returns:
            SearchExpandResult with parent documents and their content

        Example:
            result = client.search_for_parent(
                collection_name="tenant_abc_test_kb",
                query="email validation",
                expand_top_n=5,
                filters=MetadataFilter(document_type="testcase"),
                score_aggregation="mean"
            )
            for parent in result.parents:
                print(f"Test Case (score: {parent.score}):")
                print(parent.content[:200])
        """
        request_data = {
            "collection_name": collection_name,
            "query": query,
            "top_k": top_k,
            "expand_top_n": expand_top_n,
            "include_children": include_children,
            "score_aggregation": score_aggregation
        }
        if filters:
            request_data["filters"] = filters.to_dict() if hasattr(filters, 'to_dict') else filters
        if score_threshold is not None:
            request_data["score_threshold"] = score_threshold
        if search_type is not None:
            request_data["search_type"] = search_type

        data = self._make_request("POST", "/api/knowledge/search/parent", json=request_data)

        parents = []
        for p in data.get("parents", []):
            parent = ParentResult(
                parent_id=p["parent_id"],
                content=p["content"],
                score=p["score"],
                metadata=p.get("metadata", {}),
                child_count=p.get("child_count", 0)
            )
            if include_children and p.get("matching_children"):
                parent.matching_children = [SearchResult(**c) for c in p["matching_children"]]
            parents.append(parent)

        return SearchExpandResult(
            parents=parents,
            total_parents=data.get("total_parents", len(parents)),
            total_children_searched=data.get("total_children_searched", 0),
            query=data.get("query", query),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            cached=data.get("cached", False)
        )

    # =========================================================================
    # TEXT EXTRACTION OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def extract_text(
        self,
        file_content: bytes,
        filename: str
    ) -> ExtractionResult:
        """
        Extract text content from a document file.

        Args:
            file_content: File content as bytes
            filename: Original filename (used for format detection)

        Returns:
            ExtractionResult with extracted text content

        Raises:
            KnowledgeValidationError: If file format is not supported
            KnowledgeAPIError: If extraction fails
        """
        try:
            client = self._get_client()

            # Apply interceptors to headers (without Content-Type for multipart)
            headers = {}
            headers = self._apply_interceptors(headers)

            # Send file as multipart form data
            files = {"file": (filename, file_content)}
            response = client.post(
                "/api/knowledge/extraction/extract",
                files=files,
                headers=headers
            )
            data = self._handle_response(response)
            return ExtractionResult(**data)
        except httpx.ConnectError as e:
            raise KnowledgeConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise KnowledgeTimeoutError(f"Request timed out: {e}")

    @retry_with_backoff(max_retries=3)
    def get_supported_formats(self) -> SupportedFormats:
        """
        Get list of supported file formats for text extraction.

        Returns:
            SupportedFormats with list of supported file extensions
        """
        data = self._make_request("GET", "/api/knowledge/extraction/formats")
        return SupportedFormats(**data)

    def is_format_supported(self, filename: str) -> bool:
        """
        Check if a file format is supported for text extraction.

        Args:
            filename: Filename to check (only extension is used)

        Returns:
            True if format is supported, False otherwise
        """
        data = self._make_request(
            "POST",
            "/api/knowledge/extraction/check-format",
            params={"filename": filename}
        )
        return data.get("supported", False)

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check knowledge-base-service health."""
        try:
            client = self._get_client()
            response = client.get("/health")
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
_knowledge_client: Optional[KnowledgeClient] = None


def get_knowledge_client() -> KnowledgeClient:
    """Get singleton KnowledgeClient instance."""
    global _knowledge_client
    if _knowledge_client is None:
        _knowledge_client = KnowledgeClient()
    return _knowledge_client
