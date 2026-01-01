from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Dict, Any

from app.models.schemas import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    ChatRequest,
    ChatResponse,
    HealthCheckResponse
)
from app.services.rag_service import RAGService
from config.settings import settings

router = APIRouter()

# Initialize RAG service lazily
rag_service = None

def get_rag_service():
    """Get or initialize RAG service"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Enterprise Policy Chatbot API",
        "version": settings.app_version,
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check service stats
        stats = get_rag_service().get_stats()

        return HealthCheckResponse(
            status="healthy",
            app_name=settings.app_name,
            version=settings.app_version,
            timestamp=datetime.now(),
            services={
                "vector_store": "operational",
                "knowledge_graph": "operational",
                "llm": "operational",
                "documents_indexed": str(stats.get("total_documents_indexed", 0)),
                "entities_extracted": str(stats.get("total_entities", 0))
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest):
    """
    Upload and process a policy document

    This endpoint:
    1. Extracts text from PDF
    2. Chunks the document
    3. Creates embeddings and stores in vector DB
    4. Extracts entities and stores in knowledge graph
    """
    try:
        # Process the document
        result = get_rag_service().process_document(
            file_path=request.file_path,
            document_type=request.document_type,
            metadata=request.metadata
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process document: {result.get('error')}"
            )

        return DocumentUploadResponse(
            document_id=result["document_id"],
            file_name=result["file_name"],
            status="processed",
            chunks_created=result["chunks_created"],
            entities_extracted=result["entities_extracted"],
            message="Document successfully processed and indexed"
        )

    except HTTPException:
        # Re-raise HTTPException without modification
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for querying policy documents

    This endpoint:
    1. Performs vector similarity search
    2. Queries knowledge graph for related entities
    3. Generates answer using LLM
    4. Returns answer with source references
    """
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        # Query the RAG service
        response = get_rag_service().query(
            question=request.query,
            session_id=request.session_id
        )

        return response

    except HTTPException:
        # Re-raise HTTPException without modification
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get system statistics"""
    try:
        stats = get_rag_service().get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stats: {str(e)}"
        )
