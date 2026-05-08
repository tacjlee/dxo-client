"""
GraphKnowledgeClient

Synchronous HTTP client for workflow-graph-knowledge service.
Provides generic graph operations for any FalkorDB-backed graph.

Usage:
    from dxo_client import GraphKnowledgeClient

    client = GraphKnowledgeClient()

    # Create nodes
    client.create_node("Viewpoint", "VP-001", {"name": "Must", "description": "Required field"})

    # Create relationships
    client.create_relationship("TestCase", "TC-001", "Viewpoint", "VP-001", "TESTS")

    # Query
    result = client.query("MATCH (n:Viewpoint) RETURN n LIMIT 10")

    # Get stats
    stats = client.get_stats()
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, Callable
from functools import wraps

import httpx

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


class GraphKnowledgeError(Exception):
    """Base exception for graph knowledge client errors."""
    pass


class GraphKnowledgeConnectionError(GraphKnowledgeError):
    """Raised when connection to graph knowledge service fails."""
    pass


class GraphKnowledgeTimeoutError(GraphKnowledgeError):
    """Raised when request to graph knowledge service times out."""
    pass


class GraphKnowledgeAPIError(GraphKnowledgeError):
    """Raised when graph knowledge service returns an error response."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GraphKnowledgeNotFoundError(GraphKnowledgeError):
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
                except (GraphKnowledgeConnectionError, GraphKnowledgeTimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class GraphKnowledgeClient:
    """
    Synchronous HTTP client for workflow-graph-knowledge service.

    Features:
    - Generic node/relationship CRUD operations
    - Cypher query execution
    - Connection pooling
    - Retry with exponential backoff
    - Request interceptors
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        read_timeout: float = 30.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        interceptors: Optional[List[Callable[[Dict[str, str]], Dict[str, str]]]] = None
    ):
        """
        Initialize GraphKnowledgeClient.

        Args:
            base_url: Service URL (default: from GRAPH_KNOWLEDGE_SERVICE_URL env or http://localhost:8006)
            read_timeout: Read timeout in seconds (default: 30)
            connect_timeout: Connection timeout in seconds (default: 10)
            max_retries: Maximum retry attempts
            interceptors: List of request interceptors
        """
        self._base_url = base_url or _get_config("GRAPH_KNOWLEDGE_SERVICE_URL", "http://localhost:8006")
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
            raise GraphKnowledgeNotFoundError(f"Resource not found: {response.text}")
        elif response.status_code >= 500:
            raise GraphKnowledgeAPIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text
            )
        else:
            raise GraphKnowledgeAPIError(
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
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            client = self._get_client()

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
            raise GraphKnowledgeConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise GraphKnowledgeTimeoutError(f"Request timed out: {e}")

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check graph knowledge service health."""
        try:
            client = self._get_client()
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def create_node(
        self,
        label: str,
        node_id: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Create or merge a node.

        Args:
            label: Node label (e.g., "Viewpoint", "TestCase")
            node_id: Unique identifier
            properties: Additional properties

        Returns:
            Created node data
        """
        return self._request(
            "POST",
            "/api/v1/graph/nodes",
            json={
                "label": label,
                "id": node_id,
                "properties": properties or {},
            },
        )

    @retry_with_backoff(max_retries=3)
    def create_nodes_bulk(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Bulk create nodes.

        Args:
            nodes: List of {"label": str, "id": str, "properties": dict}

        Returns:
            {"created": int, "errors": list}
        """
        return self._request(
            "POST",
            "/api/v1/graph/nodes/bulk",
            json={"nodes": nodes},
        )

    @retry_with_backoff(max_retries=3)
    def merge_node(
        self,
        label: str,
        node_id: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Merge a node (create if not exists, update if exists).

        This is idempotent - safe to call multiple times with the same ID.

        Args:
            label: Node label (e.g., "FlowPattern", "FlowState")
            node_id: Unique identifier
            properties: Additional properties

        Returns:
            Merged node data
        """
        return self._request(
            "POST",
            "/api/v1/graph/nodes/merge",
            json={
                "label": label,
                "id": node_id,
                "properties": properties or {},
            },
        )

    @retry_with_backoff(max_retries=3)
    def get_node(self, label: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by label and ID."""
        try:
            return self._request("GET", f"/api/v1/graph/nodes/{label}/{node_id}")
        except GraphKnowledgeNotFoundError:
            return None

    @retry_with_backoff(max_retries=3)
    def list_nodes(
        self,
        label: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List nodes by label."""
        return self._request(
            "GET",
            f"/api/v1/graph/nodes/{label}",
            params={"limit": limit, "offset": offset},
        )

    @retry_with_backoff(max_retries=3)
    def delete_node(self, label: str, node_id: str) -> bool:
        """Delete a node and its relationships."""
        try:
            self._request("DELETE", f"/api/v1/graph/nodes/{label}/{node_id}")
            return True
        except GraphKnowledgeNotFoundError:
            return False

    @retry_with_backoff(max_retries=3)
    def delete_all_nodes(self, label: str) -> int:
        """Delete all nodes with a given label."""
        result = self._request("DELETE", f"/api/v1/graph/nodes/{label}")
        return result.get("deleted", 0)

    # =========================================================================
    # RELATIONSHIP OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def create_relationship(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            from_label: Source node label
            from_id: Source node ID
            to_label: Target node label
            to_id: Target node ID
            rel_type: Relationship type (e.g., "TESTS", "BELONGS_TO")
            properties: Additional properties

        Returns:
            Created relationship data
        """
        return self._request(
            "POST",
            "/api/v1/graph/relationships",
            json={
                "from_label": from_label,
                "from_id": from_id,
                "to_label": to_label,
                "to_id": to_id,
                "type": rel_type,
                "properties": properties or {},
            },
        )

    @retry_with_backoff(max_retries=3)
    def create_relationships_bulk(
        self,
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Bulk create relationships.

        Args:
            relationships: List of relationship dicts

        Returns:
            {"created": int, "errors": list}
        """
        return self._request(
            "POST",
            "/api/v1/graph/relationships/bulk",
            json={"relationships": relationships},
        )

    @retry_with_backoff(max_retries=3)
    def merge_relationship(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Merge a relationship (create if not exists, idempotent).

        This is idempotent - safe to call multiple times.

        Args:
            from_label: Source node label
            from_id: Source node ID
            to_label: Target node label
            to_id: Target node ID
            rel_type: Relationship type (e.g., "HAS_STATE", "HAS_ACTION")
            properties: Additional properties

        Returns:
            Merged relationship data
        """
        return self._request(
            "POST",
            "/api/v1/graph/relationships/merge",
            json={
                "from_label": from_label,
                "from_id": from_id,
                "to_label": to_label,
                "to_id": to_id,
                "type": rel_type,
                "properties": properties or {},
            },
        )

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def query(
        self,
        cypher: str,
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query.

        Args:
            cypher: The Cypher query
            parameters: Query parameters

        Returns:
            {"columns": list, "rows": list, "stats": dict}
        """
        return self._request(
            "POST",
            "/api/v1/graph/query",
            json={
                "cypher": cypher,
                "parameters": parameters or {},
            },
        )

    # =========================================================================
    # STATS OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            {"graph_name": str, "total_nodes": int, "total_relationships": int, ...}
        """
        return self._request("GET", "/api/v1/graph/stats")

    # =========================================================================
    # INDEX OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def create_index(self, label: str, property: str) -> Dict[str, Any]:
        """Create an index on a node label and property."""
        return self._request(
            "POST",
            "/api/v1/graph/indexes",
            json={"label": label, "property": property},
        )

    # =========================================================================
    # ADMIN OPERATIONS
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def delete_all(self, confirm: bool = False) -> Dict[str, Any]:
        """Delete all data from the graph. Requires confirm=True."""
        return self._request(
            "DELETE",
            "/api/v1/graph/all",
            params={"confirm": confirm},
        )


# Singleton instance
_graph_knowledge_client: Optional[GraphKnowledgeClient] = None


def get_graph_knowledge_client() -> GraphKnowledgeClient:
    """Get singleton GraphKnowledgeClient instance."""
    global _graph_knowledge_client
    if _graph_knowledge_client is None:
        _graph_knowledge_client = GraphKnowledgeClient()
    return _graph_knowledge_client
