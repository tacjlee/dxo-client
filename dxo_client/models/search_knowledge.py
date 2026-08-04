"""
SearchKnowledgeClient Models

Pydantic models for the dxo-search-knowledge service (hybrid search, push-API).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SearchChunk(BaseModel):
    """
    One pre-cut chunk to be indexed.

    chunk_id must be a UUID string if provided (Qdrant point ID constraint).
    Omit it to let the service derive a stable ID from hash(doc_id + text),
    which makes ingestion idempotent.
    """
    text: str = Field(min_length=1)
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestChunksResult(BaseModel):
    """Result of indexing a batch of chunks for one document."""
    doc_id: str
    chunks_indexed: int
    chunk_ids: List[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    """One hit returned by hybrid search."""
    chunk_id: str
    doc_id: str
    score: float
    text: str
    source: str  # which retriever/fusion produced the hit (e.g. "semantic", "bm25", "rrf")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchQueryResult(BaseModel):
    """Response of a hybrid search query."""
    query: str
    hits: List[SearchHit] = Field(default_factory=list)

    @property
    def texts(self) -> List[str]:
        """Convenience accessor: hit texts in rank order."""
        return [hit.text for hit in self.hits]

    @property
    def combined_context(self) -> str:
        """Hit texts joined for direct use as RAG context."""
        return "\n\n".join(self.texts)
