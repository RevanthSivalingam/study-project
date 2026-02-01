from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """Request model for uploading documents"""
    file_path: str = Field(..., description="Path to the PDF file")
    document_type: str = Field(default="policy", description="Type of document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response model after document upload"""
    document_id: str
    file_name: str
    status: str
    chunks_created: int
    entities_extracted: int
    message: str


class ChatRequest(BaseModel):
    """Request model for chat queries"""
    query: str = Field(..., description="User's question")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")


class SourceReference(BaseModel):
    """Source document reference"""
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float
    excerpt: str = Field(..., description="Relevant excerpt from the document")


class ChatResponse(BaseModel):
    """Response model for chat queries"""
    answer: str = Field(..., description="The chatbot's answer")
    sources: List[SourceReference] = Field(..., description="Source references")
    confidence_score: Optional[float] = Field(default=None)
    entities_found: Optional[List[str]] = Field(default=None)

    # Enhanced RAG fields (optional, backward compatible)
    retrieval_method: Optional[str] = Field(
        default=None,
        description="Retrieval method used: 'kg_guided', 'semantic_fallback', or 'fixed_chunk'"
    )
    mmr_sentences_used: Optional[int] = Field(
        default=None,
        description="Number of sentences selected via MMR"
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Cluster ID used for semantic fallback"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Title of the section retrieved from"
    )
    precision_at_k: Optional[float] = Field(
        default=None,
        description="Precision@k evaluation metric"
    )
    recall_at_k: Optional[float] = Field(
        default=None,
        description="Recall@k evaluation metric"
    )
    mrr: Optional[float] = Field(
        default=None,
        description="Mean Reciprocal Rank evaluation metric"
    )


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
