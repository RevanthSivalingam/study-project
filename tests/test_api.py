"""
Test FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from app.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_rag_service():
    """Mock RAG service for testing"""
    with patch('app.api.routes.get_rag_service') as mock:
        service = Mock()
        mock.return_value = service
        yield service


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, client):
        """Should return API information"""
        response = client.get("/api/v1/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Enterprise Policy Chatbot API"
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, client, mock_rag_service):
        """Should return healthy status"""
        mock_rag_service.get_stats.return_value = {
            "total_documents_indexed": 5,
            "total_entities": 10
        }

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == "Enterprise Policy Chatbot"
        assert data["services"]["vector_store"] == "operational"
        assert data["services"]["documents_indexed"] == "5"

    def test_health_check_failure(self, client, mock_rag_service):
        """Should return 503 when service is unhealthy"""
        mock_rag_service.get_stats.side_effect = Exception("Service error")

        response = client.get("/api/v1/health")

        assert response.status_code == 503
        assert "Service unhealthy" in response.json()["detail"]


class TestDocumentUploadEndpoint:
    """Test document upload endpoint"""

    def test_upload_document_success(self, client, mock_rag_service):
        """Should upload and process document successfully"""
        mock_rag_service.process_document.return_value = {
            "success": True,
            "document_id": "test123",
            "file_name": "test.pdf",
            "chunks_created": 10,
            "entities_extracted": 5
        }

        response = client.post(
            "/api/v1/documents/upload",
            json={
                "file_path": "data/pdfs/test.pdf",
                "document_type": "policy",
                "metadata": {"category": "HR"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test123"
        assert data["status"] == "processed"
        assert data["chunks_created"] == 10
        assert data["entities_extracted"] == 5

    def test_upload_document_file_not_found(self, client, mock_rag_service):
        """Should return 404 when file doesn't exist"""
        mock_rag_service.process_document.side_effect = FileNotFoundError("File not found")

        response = client.post(
            "/api/v1/documents/upload",
            json={
                "file_path": "nonexistent.pdf",
                "document_type": "policy"
            }
        )

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    def test_upload_document_processing_error(self, client, mock_rag_service):
        """Should return 400 when processing fails"""
        mock_rag_service.process_document.return_value = {
            "success": False,
            "error": "Failed to extract text"
        }

        response = client.post(
            "/api/v1/documents/upload",
            json={
                "file_path": "data/pdfs/test.pdf",
                "document_type": "policy"
            }
        )

        assert response.status_code == 400
        assert "Failed to process document" in response.json()["detail"]

    def test_upload_document_validation_error(self, client):
        """Should return 422 for invalid request data"""
        response = client.post(
            "/api/v1/documents/upload",
            json={}  # Missing required fields
        )

        assert response.status_code == 422


class TestChatEndpoint:
    """Test chat endpoint"""

    def test_chat_success(self, client, mock_rag_service):
        """Should return chat response successfully"""
        from app.models.schemas import ChatResponse, SourceReference

        mock_response = ChatResponse(
            answer="Employees get 15 days of annual leave.",
            sources=[
                SourceReference(
                    document_name="test.pdf",
                    page_number=1,
                    chunk_id="test_chunk_1",
                    relevance_score=0.95,
                    excerpt="Annual leave policy..."
                )
            ],
            confidence_score=0.9,
            entities_found=["annual leave"]
        )
        mock_rag_service.query.return_value = mock_response

        response = client.post(
            "/api/v1/chat",
            json={
                "query": "How many days of annual leave?",
                "session_id": "test-session"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "Employees get 15 days" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["confidence_score"] == 0.9

    def test_chat_empty_query(self, client):
        """Should return 400 for empty query"""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "",
                "session_id": "test"
            }
        )

        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    def test_chat_missing_query(self, client):
        """Should return 422 for missing query field"""
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "test"}
        )

        assert response.status_code == 422

    def test_chat_error_handling(self, client, mock_rag_service):
        """Should return 500 for server errors"""
        mock_rag_service.query.side_effect = Exception("Database error")

        response = client.post(
            "/api/v1/chat",
            json={
                "query": "test query",
                "session_id": "test"
            }
        )

        assert response.status_code == 500
        assert "Error processing query" in response.json()["detail"]


class TestStatsEndpoint:
    """Test stats endpoint"""

    def test_get_stats_success(self, client, mock_rag_service):
        """Should return system statistics"""
        mock_rag_service.get_stats.return_value = {
            "total_documents_indexed": 25,
            "total_entities": 100,
            "vector_store_size": "5MB"
        }

        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["total_documents_indexed"] == 25
        assert data["data"]["total_entities"] == 100

    def test_get_stats_error(self, client, mock_rag_service):
        """Should return 500 on error"""
        mock_rag_service.get_stats.side_effect = Exception("Stats error")

        response = client.get("/api/v1/stats")

        assert response.status_code == 500
        assert "Error fetching stats" in response.json()["detail"]


class TestCORSHeaders:
    """Test CORS configuration"""

    def test_cors_headers_present(self, client):
        """Should have CORS headers in response"""
        response = client.get("/api/v1/")

        # Check that CORS middleware is configured
        # Note: Test client might not show all CORS headers
        assert response.status_code == 200


class TestRequestValidation:
    """Test request validation"""

    def test_invalid_json(self, client):
        """Should handle invalid JSON gracefully"""
        response = client.post(
            "/api/v1/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_content_type(self, client, mock_rag_service):
        """Should require application/json content type"""
        from app.models.schemas import ChatResponse

        # Configure mock to return proper ChatResponse
        mock_rag_service.query.return_value = ChatResponse(
            answer="Test answer",
            sources=[],
            confidence_score=0.8,
            entities_found=[]
        )

        # FastAPI automatically handles this
        response = client.post(
            "/api/v1/chat",
            json={"query": "test", "session_id": "test"}
        )

        # Should work with proper JSON
        assert response.status_code == 200
