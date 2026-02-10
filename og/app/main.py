"""
FastAPI Backend for RAG Chatbot
Provides REST API endpoints for document management and querying
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_logic import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="REST API for document-based question answering using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None
system_initialized = False

# Request/Response Models
class DocumentUploadRequest(BaseModel):
    file_path: str
    document_type: Optional[str] = "policy"
    metadata: Optional[Dict[str, Any]] = {}

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_created: Optional[int] = None
    entities_extracted: Optional[int] = None

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = []
    confidence: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = 0.0  # Deprecated, kept for backward compatibility
    entities_found: Optional[List[str]] = []
    method: Optional[str] = None
    section_title: Optional[str] = None
    llm_provider: Optional[str] = None
    tokens_used: Optional[int] = None
    fallback_used: Optional[bool] = None

class StatsResponse(BaseModel):
    data: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    initialized: bool

class InitializeRequest(BaseModel):
    api_key: Optional[str] = None
    input_folder: Optional[str] = "inputfiles"
    provider: Optional[str] = "none"
    gemini_api_key: Optional[str] = None

class InitializeResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, Any]] = None


# API Endpoints

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Check API health and initialization status"""
    return {
        "status": "healthy",
        "initialized": system_initialized
    }

@app.post("/api/v1/initialize", response_model=InitializeResponse)
async def initialize_system(request: InitializeRequest):
    """Initialize the RAG system"""
    global rag_system, system_initialized

    try:
        # Build LLM config from request
        llm_config = {
            'provider': request.provider,
            'openai_api_key': request.api_key,
            'gemini_api_key': request.gemini_api_key
        }

        # Create RAG system
        rag_system = RAGSystem(
            input_folder=request.input_folder,
            llm_config=llm_config
        )

        # Initialize
        success, message = rag_system.initialize()

        if success:
            system_initialized = True
            stats = {
                "documents": len(rag_system.documents),
                "sections": len(rag_system.sections),
                "clusters": rag_system.NUM_CLUSTERS,
                "learned_terms": len(rag_system.learned_terms),
                "kg_entities": len(rag_system.knowledge_graph)
            }
            return {
                "success": True,
                "message": message,
                "stats": stats
            }
        else:
            system_initialized = False
            return {
                "success": False,
                "message": message,
                "stats": None
            }

    except Exception as e:
        system_initialized = False
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest):
    """Upload and process a document"""
    global rag_system, system_initialized

    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        # Initialize or reinitialize system
        input_folder = str(Path(request.file_path).parent)

        rag_system = RAGSystem(input_folder=input_folder)
        success, message = rag_system.initialize()

        if success:
            system_initialized = True

            return {
                "success": True,
                "message": message,
                "document_id": Path(request.file_path).stem,
                "chunks_created": len(rag_system.sections),
                "entities_extracted": len(rag_system.knowledge_graph)
            }
        else:
            raise HTTPException(status_code=500, detail=message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query"""
    global rag_system, system_initialized

    if not system_initialized or rag_system is None:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please upload documents first or call /api/v1/initialize"
        )

    try:
        # Process query
        result = rag_system.answer_query(request.query)

        # Use actual sources from result or format from retrieved sentences
        sources = result.get("sources", [])
        if not sources and result.get("retrieved_sentences"):
            # Fallback for backward compatibility
            sources = []
            for idx, sentence in enumerate(result["retrieved_sentences"], 1):
                sources.append({
                    "text": sentence,
                    "section": result.get("section_title", "Unknown"),
                    "relevance_score": 0.7,
                    "rank": idx
                })

        # Get confidence object (new format) or create from score (old format)
        confidence_data = result.get("confidence", {})
        confidence_score = confidence_data.get("score", 0.0) if confidence_data else 0.7

        return {
            "answer": result.get("answer", "No answer generated"),
            "sources": sources,
            "confidence": confidence_data,
            "confidence_score": confidence_score,  # Backward compatibility
            "entities_found": [],
            "method": result.get("method"),
            "section_title": result.get("section_title"),
            "llm_provider": result.get("llm_provider"),
            "tokens_used": result.get("tokens_used", 0),
            "fallback_used": result.get("fallback_used", False)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    global rag_system, system_initialized

    if not system_initialized or rag_system is None:
        return {
            "data": {
                "total_documents_indexed": 0,
                "total_entities": 0,
                "total_sections": 0,
                "total_clusters": 0,
                "learned_terms": 0
            }
        }

    return {
        "data": {
            "total_documents_indexed": len(rag_system.documents),
            "total_entities": len(rag_system.knowledge_graph),
            "total_sections": len(rag_system.sections),
            "total_clusters": rag_system.NUM_CLUSTERS,
            "learned_terms": len(rag_system.learned_terms)
        }
    }

@app.post("/api/v1/reset")
async def reset_system():
    """Reset the system"""
    global rag_system, system_initialized

    rag_system = None
    system_initialized = False

    return {"success": True, "message": "System reset successfully"}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup if documents exist"""
    global rag_system, system_initialized

    input_folder = Path("inputfiles")
    if input_folder.exists() and any(input_folder.iterdir()):
        try:
            rag_system = RAGSystem(input_folder="inputfiles")
            success, message = rag_system.initialize()
            if success:
                system_initialized = True
                print(f"✅ System auto-initialized: {message}")
            else:
                print(f"⚠️ Auto-initialization failed: {message}")
        except Exception as e:
            print(f"⚠️ Auto-initialization error: {e}")


def main():
    """Run the FastAPI server"""
    print("=" * 60)
    print("  🚀 RAG Chatbot API Server")
    print("=" * 60)
    print()
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/api/v1/health")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
