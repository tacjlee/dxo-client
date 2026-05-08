"""
Models package for dxo_client.

This package contains shared models used across PEV microservices.
"""

# Knowledge Client Models (for backwards compatibility with models.py)
from .knowledge import (
    MetadataFilter,
    CollectionInfo,
    SearchResult,
    RAGContext,
    DocumentChunk,
    DocumentProcessResult,
    ExtractionResult,
    SupportedFormats,
    # Parent-Child Chunking Models
    ParentChildChunkConfig,
    ParentChildProcessResult,
    ParentResult,
    SearchExpandResult,
    # Similarity API Models
    SimilarityRequest,
    SimilarityResponse,
    BatchSimilarityItem,
    BatchSimilarityRequest,
    BatchSimilarityResult,
    BatchSimilarityResponse,
    # Search Records API Models
    RecordMatch,
    SearchRecordsResponse,
)

__all__ = [
    # Knowledge Client Models
    "MetadataFilter",
    "CollectionInfo",
    "SearchResult",
    "RAGContext",
    "DocumentChunk",
    "DocumentProcessResult",
    "ExtractionResult",
    "SupportedFormats",
    # Parent-Child Chunking Models
    "ParentChildChunkConfig",
    "ParentChildProcessResult",
    "ParentResult",
    "SearchExpandResult",
    # Similarity API Models
    "SimilarityRequest",
    "SimilarityResponse",
    "BatchSimilarityItem",
    "BatchSimilarityRequest",
    "BatchSimilarityResult",
    "BatchSimilarityResponse",
    # Search Records API Models
    "RecordMatch",
    "SearchRecordsResponse",
]
