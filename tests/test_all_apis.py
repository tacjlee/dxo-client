"""
Comprehensive end-to-end tests for dxo-client calling ALL workflow-knowledge APIs.

This test file validates the complete KnowledgeClient functionality by exercising
every API endpoint in a realistic workflow scenario.

Install dxo-client from git:
    pip install git+https://github.com/tacjlee/dxo-client.git

Run with:
    pytest tests/test_all_apis.py -v

Run integration tests only:
    pytest tests/test_all_apis.py -v -k "Integration"

API Coverage:
    1. Health Check API
       - GET /health

    2. Collection APIs
       - POST /api/knowledge/collections (create_collection)
       - GET /api/knowledge/collections/{name} (get_collection_info)
       - GET /api/knowledge/collections (list_collections)
       - DELETE /api/knowledge/collections/{name} (delete_collection)

    3. Document APIs
       - POST /api/knowledge/documents/process (add_documents)
       - DELETE /api/knowledge/documents (delete_documents)

    4. Vector APIs
       - POST /api/knowledge/vectors (add_vectors)
       - DELETE /api/knowledge/vectors (delete_vectors)

    5. Embedding APIs
       - POST /api/knowledge/embeddings (generate_embeddings)

    6. Search APIs
       - POST /api/knowledge/search (search)
       - POST /api/knowledge/search/rag (rag_retrieval)
"""

import pytest
import os
import uuid
import time
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dxo_client import KnowledgeClient, MetadataFilter
from dxo_client.models import (
    CollectionInfo,
    SearchResult,
    RAGContext,
    DocumentProcessResult,
)
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
def client(knowledge_base_url):
    """Create a client for tests."""
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
def collection_suffix():
    """Generate unique collection suffix."""
    return f"allapi-{uuid.uuid4().hex[:6]}"


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
# TEST DATA
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "content": "Machine learning is a branch of artificial intelligence that focuses on building "
                   "applications that learn from data and improve their accuracy over time without "
                   "being explicitly programmed. ML algorithms use historical data as input to predict "
                   "new output values.",
        "metadata": {"document_id": "ml-intro", "category": "ai", "topic": "machine-learning"}
    },
    {
        "content": "Deep learning is a subset of machine learning that uses neural networks with "
                   "many layers (deep neural networks) to analyze various factors of data. Deep learning "
                   "enables the analysis of unstructured data such as images, audio, and text.",
        "metadata": {"document_id": "dl-intro", "category": "ai", "topic": "deep-learning"}
    },
    {
        "content": "Natural Language Processing (NLP) is a field of AI that gives machines the ability "
                   "to read, understand, and derive meaning from human languages. NLP combines "
                   "computational linguistics with statistical and machine learning models.",
        "metadata": {"document_id": "nlp-intro", "category": "ai", "topic": "nlp"}
    },
    {
        "content": "Vector databases are specialized databases designed to store, index, and query "
                   "high-dimensional vectors efficiently. They are essential for similarity search "
                   "applications, recommendation systems, and AI/ML workloads.",
        "metadata": {"document_id": "vecdb-intro", "category": "database", "topic": "vector-db"}
    },
    {
        "content": "Retrieval Augmented Generation (RAG) is an AI framework that combines information "
                   "retrieval with generative AI models. RAG retrieves relevant documents from a "
                   "knowledge base and uses them as context for generating accurate responses.",
        "metadata": {"document_id": "rag-intro", "category": "ai", "topic": "rag"}
    },
]


# ============================================================================
# COMPREHENSIVE INTEGRATION TEST
# ============================================================================

@requires_service
class TestAllAPIsIntegration:
    """
    Comprehensive integration tests that exercise ALL workflow-knowledge APIs.

    Test workflow:
    1. Health Check
    2. Create Collection
    3. Get Collection Info
    4. List Collections
    5. Generate Embeddings
    6. Add Vectors
    7. Add Documents
    8. Similarity Search
    9. RAG Retrieval
    10. Delete Documents
    11. Delete Vectors
    12. Delete Collection
    """

    @pytest.fixture(autouse=True)
    def setup_test_environment(
        self, client, test_tenant_id, test_knowledge_id, collection_suffix
    ):
        """Set up test environment and clean up after."""
        self.client = client
        self.tenant_id = test_tenant_id
        self.knowledge_id = test_knowledge_id
        self.collection_suffix = collection_suffix
        self.sanitized_tenant = test_tenant_id.replace('-', '_')
        self.collection_name = f"tenant_{self.sanitized_tenant}_{collection_suffix}"

        yield

        # Cleanup: delete collection if exists
        try:
            self.client.delete_collection(self.collection_name, force=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # API 1: Health Check
    # -------------------------------------------------------------------------

    def test_01_health_check(self):
        """Test health check API - GET /health"""
        health = self.client.health_check()

        assert health["status"] == "healthy"
        assert "data" in health
        print(f"  Health check: {health['status']}")

    # -------------------------------------------------------------------------
    # API 2-4: Collection APIs
    # -------------------------------------------------------------------------

    def test_02_create_collection(self):
        """Test create collection API - POST /api/knowledge/collections"""
        result = self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True,
            vector_size=1024
        )

        assert isinstance(result, CollectionInfo)
        assert result.name == self.collection_name
        assert result.config is not None
        print(f"  Created collection: {result.name}")

    def test_03_get_collection_info(self):
        """Test get collection info API - GET /api/knowledge/collections/{name}"""
        # Create collection first
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        info = self.client.get_collection_info(self.collection_name)

        assert isinstance(info, CollectionInfo)
        assert info.name == self.collection_name
        assert info.vectors_count >= 0
        print(f"  Collection info: {info.name}, vectors: {info.vectors_count}")

    def test_04_list_collections(self):
        """Test list collections API - GET /api/knowledge/collections"""
        # Create collection first
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix
        )

        # List all collections
        all_collections = self.client.list_collections()
        assert isinstance(all_collections, list)
        assert len(all_collections) >= 1
        assert any(c.name == self.collection_name for c in all_collections)

        # List by tenant
        tenant_collections = self.client.list_collections(tenant_id=self.tenant_id)
        assert isinstance(tenant_collections, list)
        for c in tenant_collections:
            assert self.sanitized_tenant in c.name
        print(f"  Listed {len(all_collections)} total, {len(tenant_collections)} for tenant")

    # -------------------------------------------------------------------------
    # API 5: Embedding API
    # -------------------------------------------------------------------------

    def test_05_generate_embeddings(self):
        """Test generate embeddings API - POST /api/knowledge/embeddings"""
        texts = [
            "Machine learning algorithms",
            "Natural language processing",
            "Vector similarity search"
        ]

        embeddings = self.client.generate_embeddings(texts, batch_size=16)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1024  # BGE-M3 dimension
            assert all(isinstance(v, float) for v in emb)
        print(f"  Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")

    # -------------------------------------------------------------------------
    # API 6: Vector API - Add
    # -------------------------------------------------------------------------

    def test_06_add_vectors(self):
        """Test add vectors API - POST /api/knowledge/vectors"""
        # Create collection first
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        vectors = [
            {
                "content": "Test vector 1 about machine learning.",
                "metadata": {"tenant_id": self.tenant_id, "type": "test"}
            },
            {
                "content": "Test vector 2 about deep learning.",
                "metadata": {"tenant_id": self.tenant_id, "type": "test"}
            },
        ]

        vector_ids = self.client.add_vectors(self.collection_name, vectors)

        assert len(vector_ids) == 2
        assert all(vid for vid in vector_ids)  # All non-empty
        print(f"  Added {len(vector_ids)} vectors: {vector_ids}")

    # -------------------------------------------------------------------------
    # API 7: Document API - Add
    # -------------------------------------------------------------------------

    def test_07_add_documents(self):
        """Test add documents API - POST /api/knowledge/documents/process"""
        # Create collection first
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        documents = SAMPLE_DOCUMENTS[:3]

        result = self.client.add_documents(
            collection_name=self.collection_name,
            documents=documents,
            tenant_id=self.tenant_id,
            knowledge_id=self.knowledge_id,
            chunk_size=500,
            chunk_overlap=50
        )

        assert isinstance(result, DocumentProcessResult)
        assert result.chunks_count >= 3
        assert result.status == "processed"
        print(f"  Processed {len(documents)} docs, {result.chunks_count} chunks")

    # -------------------------------------------------------------------------
    # API 8: Search API - Similarity
    # -------------------------------------------------------------------------

    def test_08_search(self):
        """Test similarity search API - POST /api/knowledge/search"""
        # Setup: create collection and add documents
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        # Add vectors for search
        vectors = [
            {"content": doc["content"], "metadata": {**doc["metadata"], "tenant_id": self.tenant_id}}
            for doc in SAMPLE_DOCUMENTS
        ]
        self.client.add_vectors(self.collection_name, vectors)
        time.sleep(1)  # Wait for indexing

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query="What is machine learning and artificial intelligence?",
            top_k=5,
            score_threshold=0.0
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # Results should be ordered by score
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
        print(f"  Search returned {len(results)} results, top score: {results[0].score:.3f}")

    def test_08b_search_with_filters(self):
        """Test similarity search with metadata filters."""
        # Setup: create collection and add documents
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        vectors = [
            {"content": doc["content"], "metadata": {**doc["metadata"], "tenant_id": self.tenant_id}}
            for doc in SAMPLE_DOCUMENTS
        ]
        self.client.add_vectors(self.collection_name, vectors)
        time.sleep(1)

        # Search with filter
        filters = MetadataFilter(tenant_id=self.tenant_id)
        results = self.client.search(
            collection_name=self.collection_name,
            query="neural networks",
            top_k=3,
            filters=filters
        )

        assert isinstance(results, list)
        print(f"  Filtered search returned {len(results)} results")

    # -------------------------------------------------------------------------
    # API 9: Search API - RAG
    # -------------------------------------------------------------------------

    def test_09_rag_retrieval(self):
        """Test RAG retrieval API - POST /api/knowledge/search/rag"""
        # Setup: create collection and add documents
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        vectors = [
            {"content": doc["content"], "metadata": {**doc["metadata"], "tenant_id": self.tenant_id}}
            for doc in SAMPLE_DOCUMENTS
        ]
        self.client.add_vectors(self.collection_name, vectors)
        time.sleep(1)

        # RAG retrieval
        result = self.client.rag_retrieval(
            collection_name=self.collection_name,
            query="Explain retrieval augmented generation and how it works.",
            top_k=3
        )

        assert isinstance(result, RAGContext)
        assert len(result.chunks) <= 3
        assert isinstance(result.combined_context, str)
        assert isinstance(result.source_documents, list)
        print(f"  RAG returned {len(result.chunks)} chunks, context length: {len(result.combined_context)}")

    def test_09b_rag_retrieval_with_filters(self):
        """Test RAG retrieval with metadata filters."""
        # Setup
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        vectors = [
            {"content": doc["content"], "metadata": {**doc["metadata"], "tenant_id": self.tenant_id}}
            for doc in SAMPLE_DOCUMENTS
        ]
        self.client.add_vectors(self.collection_name, vectors)
        time.sleep(1)

        # RAG with filter
        filters = MetadataFilter(tenant_id=self.tenant_id)
        result = self.client.rag_retrieval(
            collection_name=self.collection_name,
            query="What are vector databases?",
            top_k=3,
            filters=filters
        )

        assert isinstance(result, RAGContext)
        print(f"  Filtered RAG returned {len(result.chunks)} chunks")

    # -------------------------------------------------------------------------
    # API 10: Document API - Delete
    # -------------------------------------------------------------------------

    def test_10_delete_documents(self):
        """Test delete documents API - DELETE /api/knowledge/documents"""
        # Setup
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        documents = [
            {"content": "Document to delete.", "metadata": {"document_id": "delete-me"}}
        ]
        self.client.add_documents(
            collection_name=self.collection_name,
            documents=documents,
            tenant_id=self.tenant_id,
            knowledge_id=self.knowledge_id
        )
        time.sleep(1)

        # Delete
        deleted_count = self.client.delete_documents(
            collection_name=self.collection_name,
            document_id="delete-me"
        )

        # Filter-based deletes return -1 (Qdrant doesn't provide count)
        assert deleted_count == -1 or deleted_count >= 0
        print(f"  Deleted {deleted_count} document chunks")

    # -------------------------------------------------------------------------
    # API 11: Vector API - Delete
    # -------------------------------------------------------------------------

    def test_11_delete_vectors(self):
        """Test delete vectors API - DELETE /api/knowledge/vectors"""
        # Setup
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix,
            enable_multivector=True
        )

        vectors = [
            {"content": "Vector 1 to delete.", "metadata": {"batch": "delete-batch"}},
            {"content": "Vector 2 to delete.", "metadata": {"batch": "delete-batch"}},
        ]
        vector_ids = self.client.add_vectors(self.collection_name, vectors)
        time.sleep(1)

        # Delete by IDs
        deleted_count = self.client.delete_vectors(
            collection_name=self.collection_name,
            vector_ids=vector_ids
        )

        assert deleted_count >= 0
        print(f"  Deleted {deleted_count} vectors")

    # -------------------------------------------------------------------------
    # API 12: Collection API - Delete
    # -------------------------------------------------------------------------

    def test_12_delete_collection(self):
        """Test delete collection API - DELETE /api/knowledge/collections/{name}"""
        # Create collection
        self.client.create_collection(
            tenant_id=self.tenant_id,
            name=self.collection_suffix
        )

        # Delete
        result = self.client.delete_collection(self.collection_name, force=True)

        assert result is True

        # Verify deleted
        with pytest.raises(KnowledgeNotFoundError):
            self.client.get_collection_info(self.collection_name)
        print(f"  Deleted collection: {self.collection_name}")


# ============================================================================
# FULL WORKFLOW TEST
# ============================================================================

@requires_service
class TestCompleteWorkflow:
    """
    Test complete RAG workflow using all APIs in sequence.

    This simulates a real-world usage pattern:
    1. Create knowledge base collection
    2. Ingest documents
    3. Search and retrieve
    4. Clean up
    """

    def test_complete_rag_workflow(self, client):
        """Test complete RAG workflow from creation to cleanup."""
        # Generate unique identifiers
        tenant_id = f"workflow-{uuid.uuid4().hex[:8]}"
        knowledge_id = f"kb-{uuid.uuid4().hex[:8]}"
        sanitized = tenant_id.replace('-', '_')
        collection_name = f"tenant_{sanitized}_workflow"

        print("\n=== Complete RAG Workflow Test ===")

        try:
            # Step 1: Health Check
            print("\n1. Checking service health...")
            health = client.health_check()
            assert health["status"] == "healthy"
            print(f"   Service is {health['status']}")

            # Step 2: Create Collection
            print("\n2. Creating collection...")
            collection = client.create_collection(
                tenant_id=tenant_id,
                name="workflow",
                enable_multivector=True,
                vector_size=1024
            )
            assert collection.name == collection_name
            print(f"   Created: {collection.name}")

            # Step 3: Verify Collection
            print("\n3. Verifying collection...")
            info = client.get_collection_info(collection_name)
            assert info.name == collection_name
            print(f"   Verified: {info.name}, status: {info.status}")

            # Step 4: Add Documents
            print("\n4. Adding documents...")
            result = client.add_documents(
                collection_name=collection_name,
                documents=SAMPLE_DOCUMENTS,
                tenant_id=tenant_id,
                knowledge_id=knowledge_id
            )
            assert result.status == "processed"
            print(f"   Added {len(SAMPLE_DOCUMENTS)} documents, {result.chunks_count} chunks")

            # Wait for indexing
            time.sleep(2)

            # Step 5: Generate Embeddings (standalone test)
            print("\n5. Testing embedding generation...")
            embeddings = client.generate_embeddings(["test query"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1024
            print(f"   Generated embedding with dim={len(embeddings[0])}")

            # Step 6: Similarity Search
            print("\n6. Performing similarity search...")
            search_results = client.search(
                collection_name=collection_name,
                query="What is machine learning?",
                top_k=3
            )
            print(f"   Found {len(search_results)} results")
            for i, r in enumerate(search_results):
                print(f"   [{i+1}] Score: {r.score:.3f} - {r.content[:50]}...")

            # Step 7: RAG Retrieval
            print("\n7. Performing RAG retrieval...")
            rag_result = client.rag_retrieval(
                collection_name=collection_name,
                query="Explain how RAG combines retrieval with language models.",
                top_k=3
            )
            print(f"   Retrieved {len(rag_result.chunks)} chunks")
            print(f"   Context length: {len(rag_result.combined_context)} chars")
            print(f"   Source docs: {rag_result.source_documents}")

            # Step 8: Add more vectors directly
            print("\n8. Adding vectors directly...")
            vector_ids = client.add_vectors(
                collection_name,
                [{"content": "Additional vector content.", "metadata": {"type": "extra"}}]
            )
            print(f"   Added vector: {vector_ids[0]}")

            # Step 9: Search with filters
            print("\n9. Searching with metadata filter...")
            filters = MetadataFilter(tenant_id=tenant_id)
            filtered_results = client.search(
                collection_name=collection_name,
                query="artificial intelligence",
                top_k=5,
                filters=filters
            )
            print(f"   Filtered search found {len(filtered_results)} results")

            # Step 10: Delete specific document
            print("\n10. Deleting specific document...")
            deleted = client.delete_documents(
                collection_name=collection_name,
                document_id="ml-intro"
            )
            print(f"    Deleted {deleted} chunks")

            # Step 11: Delete vector
            print("\n11. Deleting vector...")
            deleted = client.delete_vectors(
                collection_name=collection_name,
                vector_ids=vector_ids
            )
            print(f"    Deleted {deleted} vectors")

            # Step 12: List collections
            print("\n12. Listing tenant collections...")
            collections = client.list_collections(tenant_id=tenant_id)
            print(f"    Found {len(collections)} collections for tenant")

            # Step 13: Cleanup
            print("\n13. Cleaning up (deleting collection)...")
            client.delete_collection(collection_name, force=True)
            print(f"    Deleted collection: {collection_name}")

            # Verify cleanup
            with pytest.raises(KnowledgeNotFoundError):
                client.get_collection_info(collection_name)
            print("    Verified collection is deleted")

            print("\n=== Workflow Complete ===")

        except Exception as e:
            # Cleanup on failure
            try:
                client.delete_collection(collection_name, force=True)
            except Exception:
                pass
            raise e


# ============================================================================
# API COVERAGE SUMMARY TEST
# ============================================================================

@requires_service
class TestAPICoverage:
    """Verify all APIs are callable and return expected types."""

    def test_api_coverage_summary(self, client):
        """Summary test showing all API endpoints covered."""
        tenant_id = f"coverage-{uuid.uuid4().hex[:8]}"
        sanitized = tenant_id.replace('-', '_')
        collection_name = f"tenant_{sanitized}_coverage"

        apis_tested = []

        try:
            # Health
            health = client.health_check()
            assert "status" in health
            apis_tested.append("GET /health")

            # Create collection
            coll = client.create_collection(tenant_id, "coverage")
            assert isinstance(coll, CollectionInfo)
            apis_tested.append("POST /api/knowledge/collections")

            # Get collection
            info = client.get_collection_info(collection_name)
            assert isinstance(info, CollectionInfo)
            apis_tested.append("GET /api/knowledge/collections/{name}")

            # List collections
            colls = client.list_collections()
            assert isinstance(colls, list)
            apis_tested.append("GET /api/knowledge/collections")

            # Generate embeddings
            embs = client.generate_embeddings(["test"])
            assert len(embs) == 1
            apis_tested.append("POST /api/knowledge/embeddings")

            # Add vectors
            vids = client.add_vectors(collection_name, [{"content": "test", "metadata": {}}])
            assert len(vids) == 1
            apis_tested.append("POST /api/knowledge/vectors")

            time.sleep(1)

            # Similarity search
            results = client.search(collection_name, "test", top_k=1)
            assert isinstance(results, list)
            apis_tested.append("POST /api/knowledge/search")

            # RAG retrieval
            rag = client.rag_retrieval(collection_name, "test", top_k=1)
            assert isinstance(rag, RAGContext)
            apis_tested.append("POST /api/knowledge/search/rag")

            # Add documents
            doc_result = client.add_documents(
                collection_name, [{"content": "doc", "metadata": {"document_id": "d1"}}],
                tenant_id, "proj", "kb"
            )
            assert isinstance(doc_result, DocumentProcessResult)
            apis_tested.append("POST /api/knowledge/documents/process")

            time.sleep(1)

            # Delete documents
            del_docs = client.delete_documents(collection_name, document_id="d1")
            assert isinstance(del_docs, int)
            apis_tested.append("DELETE /api/knowledge/documents")

            # Delete vectors
            del_vecs = client.delete_vectors(collection_name, vector_ids=vids)
            assert isinstance(del_vecs, int)
            apis_tested.append("DELETE /api/knowledge/vectors")

            # Delete collection
            deleted = client.delete_collection(collection_name, force=True)
            assert deleted is True
            apis_tested.append("DELETE /api/knowledge/collections/{name}")

        finally:
            try:
                client.delete_collection(collection_name, force=True)
            except Exception:
                pass

        print("\n" + "=" * 60)
        print("API COVERAGE SUMMARY")
        print("=" * 60)
        for api in apis_tested:
            print(f"  [x] {api}")
        print(f"\nTotal APIs tested: {len(apis_tested)}")
        print("=" * 60)

        assert len(apis_tested) == 12, f"Expected 12 APIs, tested {len(apis_tested)}"


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
