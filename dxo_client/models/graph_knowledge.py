"""
Graph Knowledge Models

Pydantic models for workflow-graph-knowledge service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =========================================================================
# VIEWPOINT MODELS
# =========================================================================

class ViewpointCreate(BaseModel):
    """Request model for creating a viewpoint."""
    name: str
    strategy: str = "TEMPLATE"
    vp_ids: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class ViewpointUpdate(BaseModel):
    """Request model for updating a viewpoint."""
    strategy: Optional[str] = None
    vp_ids: Optional[List[str]] = None
    description: Optional[str] = None


class ViewpointNode(BaseModel):
    """Response model for a viewpoint node."""
    id: str
    name: str
    strategy: str
    vp_ids: List[str]
    description: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class CoOccurrence(BaseModel):
    """Model for viewpoint co-occurrence relationship."""
    viewpoint: str
    frequency: int
    confidence: float
    screens: List[str]


class SimilarViewpoint(BaseModel):
    """Model for similar viewpoint."""
    name: str
    score: float
    method: str


# =========================================================================
# TEST CASE MODELS
# =========================================================================

class TestCaseCreate(BaseModel):
    """Request model for creating a test case."""
    content: str
    viewpoint: str
    screen_id: str
    procedure: str = ""
    expected_result: str = ""
    widget_id: Optional[str] = None
    mode: Optional[str] = None
    priority: str = "Medium"
    source_type: str = "golden"


class TestCaseNode(BaseModel):
    """Response model for a test case node."""
    id: str
    content: str
    procedure: str
    expected_result: str
    mode: Optional[str] = None
    priority: str
    source_type: str
    viewpoint: Optional[str] = None
    screen_id: Optional[str] = None
    widget_id: Optional[str] = None
    created_at: Optional[int] = None


class TestCaseSimilarity(BaseModel):
    """Model for test case similarity result."""
    id: str
    content: str
    screen_id: str
    score: float


# =========================================================================
# SCREEN MODELS
# =========================================================================

class ScreenCreate(BaseModel):
    """Request model for creating a screen."""
    id: str
    name: str
    document_type: str = "CRUD"
    modes: List[str] = Field(default_factory=lambda: ["LIST", "ADD", "EDIT"])


class ScreenNode(BaseModel):
    """Response model for a screen node."""
    id: str
    name: str
    document_type: str
    modes: List[str]
    testcase_count: int = 0
    widget_count: int = 0
    viewpoint_count: int = 0


# =========================================================================
# WIDGET MODELS
# =========================================================================

class WidgetCreate(BaseModel):
    """Request model for creating a widget."""
    id: str
    name: str
    type: str
    screen_id: str
    semantic_type: Optional[str] = None
    constraints: Optional[str] = None


class WidgetNode(BaseModel):
    """Response model for a widget node."""
    id: str
    name: str
    type: str
    semantic_type: Optional[str] = None
    constraints: Optional[str] = None
    screen_id: Optional[str] = None


# =========================================================================
# LEARNING MODELS
# =========================================================================

class CoOccurrenceDiscovery(BaseModel):
    """Model for discovered co-occurrence."""
    viewpoint1: str
    viewpoint2: str
    frequency: int


class ViewpointSuggestion(BaseModel):
    """Model for viewpoint suggestion/recommendation."""
    viewpoint: str
    confidence: float
    reason: str
    source_screens: List[str]


class GoldenWidget(BaseModel):
    """Widget data for golden learning."""
    id: str
    name: str
    type: str
    semantic_type: Optional[str] = None


class GoldenTestCase(BaseModel):
    """Test case data for golden learning."""
    content: str
    procedure: str = ""
    expected_result: str = ""
    viewpoint: str
    widget_id: Optional[str] = None
    mode: Optional[str] = None
    priority: str = "Medium"


class GoldenLearningRequest(BaseModel):
    """Request model for learning from golden data."""
    screen_id: str
    screen_name: str
    document_type: str = "CRUD"
    modes: List[str] = Field(default_factory=lambda: ["LIST", "ADD", "EDIT"])
    widgets: List[GoldenWidget] = Field(default_factory=list)
    testcases: List[GoldenTestCase] = Field(default_factory=list)


class GoldenLearningResult(BaseModel):
    """Response model for golden learning."""
    screen_created: bool
    widgets_created: int
    testcases_created: int
    viewpoints_created: int
    co_occurrences_discovered: int


# =========================================================================
# STATS MODELS
# =========================================================================

class GraphStats(BaseModel):
    """Model for graph statistics."""
    total_nodes: int
    total_relationships: int
    nodes: Dict[str, int]
    relationships: Dict[str, int]
