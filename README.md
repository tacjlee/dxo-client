# Workflow Client

Python client for `workflow-knowledge` service. Provides a FeignClient-like interface for RAG operations.

## Installation

```bash
# From NTT GitLab (internal, source of truth)
pip install "git+https://gitlab.ntte-moi.com/dxo/dxo-client.git"

# Pin a branch or tag
pip install "git+https://gitlab.ntte-moi.com/dxo/dxo-client.git@develop"

# With Consul support
pip install "dxo-client[consul] @ git+https://gitlab.ntte-moi.com/dxo/dxo-client.git"
```

**In requirements.txt:**
```
dxo-client @ git+https://gitlab.ntte-moi.com/dxo/dxo-client.git
```

## Quick Start

```python
from dxo_client import KnowledgeClient, MetadataFilter

client = KnowledgeClient()

# Create a tenant-scoped collection
client.create_collection(tenant_id="tenant-123", name="knowledge-base")
# Creates: tenant_tenant_123_knowledge_base

# Add documents with hierarchy metadata
client.add_documents(
    collection_name="tenant_tenant_123_knowledge_base",
    documents=[
        {"content": "Document content here", "metadata": {"file_name": "doc.pdf"}}
    ],
    tenant_id="tenant-123",
    knowledge_id="kb-789"
)

# Search with tenant filtering
results = client.search(
    collection_name="tenant_tenant_123_knowledge_base",
    query="search query",
    top_k=10,
    filters=MetadataFilter(tenant_id="tenant-123")
)

# RAG retrieval
context = client.rag_retrieval(
    collection_name="tenant_tenant_123_knowledge_base",
    query="What is...",
    filters=MetadataFilter(tenant_id="tenant-123", knowledge_id="kb-789")
)
print(context.combined_context)
```

## Configuration

### Service Discovery

The client discovers the knowledge base service URL in this order:

1. **Consul** (if enabled and available)
2. **Environment variable**: `KNOWLEDGE_SERVICE_URL`
3. **Default**: `http://workflow-knowledge:8000`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWLEDGE_SERVICE_URL` | `http://workflow-knowledge:8000` | Direct service URL |
| `CONSUL_ENABLED` | `true` | Enable Consul discovery |
| `CONSUL_HOST` | `localhost` | Consul host |
| `CONSUL_PORT` | `8500` | Consul port |

### Direct URL

```python
# Bypass service discovery
client = KnowledgeClient(base_url="http://localhost:8000")
```

## Data Hierarchy

```
tenant_id      -> Collection isolation
  knowledge_id -> Metadata filter
    document_id -> Metadata filter
```

## API Reference

### Collections

```python
# Create
client.create_collection(tenant_id, name, enable_multivector=True, vector_size=1024)

# Get info
client.get_collection_info(collection_name)

# List
client.list_collections(tenant_id=None)

# Delete
client.delete_collection(collection_name, tenant_id=None, force=False)
```

### Documents

```python
# Add (with chunking and embedding)
client.add_documents(
    collection_name,
    documents,
    tenant_id,
    knowledge_id,
    user_id=None,
    chunk_size=1000,
    chunk_overlap=200
)

# Delete
client.delete_documents(
    collection_name,
    tenant_id=None,
    knowledge_id=None,
    document_id=None,
    file_name=None
)
```

### Search

```python
# Search
results = client.search(
    collection_name,
    query,
    top_k=10,
    filters=MetadataFilter(...),
    score_threshold=None,
    include_embeddings=False
)

# RAG retrieval (ColBERT reranking is automatic for multivector collections)
context = client.rag_retrieval(
    collection_name,
    query,
    top_k=5,
    filters=MetadataFilter(...)
)
```

### Embeddings

```python
embeddings = client.generate_embeddings(texts, batch_size=32)
```

## SearchKnowledgeClient (dxo-search-knowledge)

Client for the `dxo-search-knowledge` hybrid search service (semantic + BM25 → RRF → rerank).
Push-API pattern: chunking happens on your side, the service only embeds + indexes.

```python
from dxo_client import SearchKnowledgeClient

# tenant_id is sent as the X-Tenant-ID header on every request
client = SearchKnowledgeClient(
    base_url="http://localhost:8080",  # or SEARCH_KNOWLEDGE_SERVICE_URL env
    tenant_id="tenant-123",
)

# Index pre-cut chunks (idempotent when chunk_id is omitted)
result = client.ingest_chunks(
    doc_id="doc-001",
    chunks=[
        {"text": "Điều khoản bảo hành 12 tháng...", "metadata": {"section": "bảo hành"}},
        "Plain strings work too",
    ],
    metadata={"file_name": "policy.pdf"},  # document-level, merged into every chunk
)
print(result.chunks_indexed, result.chunk_ids)

# Hybrid search
response = client.search(
    "thời hạn bảo hành",
    top_k=5,
    filters={"section": "bảo hành"},   # list value = match ANY, "doc_id" filters by document
    # retrievers=["semantic"],         # optional subset; default = all enabled
)
for hit in response.hits:
    print(hit.score, hit.source, hit.text)

# RAG shortcut: joined hit texts
context = client.rag_retrieval("thời hạn bảo hành", top_k=5)

# Delete a document (all of its chunks)
client.delete_document("doc-001")

# Introspection
client.health_check()       # {"service": ..., "status": "ok"}
client.readiness_check()    # {"qdrant": true, "cache": true, "status": "ready"}
client.list_strategies()    # available/enabled retrievers + fusion strategies
```

**Environment variables**: `SEARCH_KNOWLEDGE_SERVICE_URL` (default `http://localhost:8080`),
`SEARCH_KNOWLEDGE_TENANT_ID` (default tenant). Both also resolve via Consul when available.

**Errors**: `SearchKnowledgeConnectionError`, `SearchKnowledgeTimeoutError`,
`SearchKnowledgeNotFoundError`, `SearchKnowledgeAPIError` (all subclass `SearchKnowledgeError`).

## Error Handling

```python
from dxo_client import (
    KnowledgeError,
    KnowledgeConnectionError,
    KnowledgeTimeoutError,
    KnowledgeAPIError,
    KnowledgeNotFoundError,
    KnowledgeValidationError,
)

try:
    client.search(...)
except KnowledgeConnectionError:
    # Service unreachable
except KnowledgeTimeoutError:
    # Request timed out
except KnowledgeNotFoundError:
    # Collection/resource not found
except KnowledgeValidationError:
    # Invalid request
except KnowledgeAPIError as e:
    print(e.status_code, e.response_body)
```

## Testing

### Install for Development

```bash
# Clone and install in development mode
git clone https://github.com/tacjlee/dxo-client.git
cd dxo-client
pip install -e ".[dev]"
```

### Run Tests

```bash
# Set service URL (default: http://localhost:8010)
export KNOWLEDGE_SERVICE_URL=http://localhost:8010

# Run all tests
pytest tests/ -v

# Run unit tests only (no service required)
pytest tests/ -v -k "Unit"

# Run integration tests only (requires running service)
pytest tests/ -v -k "Integration"

# Run comprehensive API coverage test
pytest tests/test_all_apis.py -v -s

# Run specific test file
pytest tests/test_collection_api.py -v
pytest tests/test_document_api.py -v
pytest tests/test_embedding_api.py -v
pytest tests/test_search_api.py -v
pytest tests/test_vector_api.py -v
```

### Test Coverage

| Test File | APIs Covered |
|-----------|--------------|
| `test_collection_api.py` | create_collection, get_collection_info, list_collections, delete_collection |
| `test_document_api.py` | add_documents, delete_documents |
| `test_vector_api.py` | add_vectors, delete_vectors |
| `test_embedding_api.py` | generate_embeddings, health_check |
| `test_search_api.py` | search, rag_retrieval |
| `test_all_apis.py` | **All APIs** - comprehensive end-to-end workflow |

### Test Types

- **Unit Tests**: Mock HTTP calls, test request/response formats, no service required
- **Integration Tests**: Require running `workflow-knowledge` service, test real API calls

## License

MIT
