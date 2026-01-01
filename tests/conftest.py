"""
Shared test fixtures and configuration
"""
import pytest
import os
from unittest.mock import Mock, patch


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set both test API keys (opt-in)"""
    # Set test API keys to avoid validation errors
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_123")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data/chroma_db")


@pytest.fixture
def gemini_only_env(monkeypatch):
    """Environment with only Gemini API key"""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def openai_only_env(monkeypatch):
    """Environment with only OpenAI API key"""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")


@pytest.fixture
def no_api_keys_env(monkeypatch):
    """Environment with no API keys (for testing errors)"""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.data = [Mock(embedding=[0.1] * 1536)]
    return mock_response


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response"""
    mock_response = Mock()
    mock_response.text = "Test response"
    return {
        'embedding': [0.1] * 768
    }


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return {
        "file_path": "data/pdfs/test_policy.pdf",
        "document_type": "policy",
        "metadata": {
            "category": "HR",
            "department": "Human Resources"
        }
    }


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "query": "What is the leave policy?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing"""
    return """
    EMPLOYEE LEAVE POLICY

    Annual Leave: Employees are entitled to 15 days of annual leave per year.
    Sick Leave: 10 days of sick leave are provided annually.
    Maternity Leave: 12 weeks of paid maternity leave.

    All leave requests must be approved by the direct manager.
    """


@pytest.fixture
def mock_vector_store():
    """Mock vector store"""
    with patch('app.services.vector_store.VectorStore') as mock:
        store = Mock()
        store.add_documents.return_value = ["doc1", "doc2"]
        store.similarity_search_with_score.return_value = [
            (Mock(page_content="test", metadata={"file_name": "test.pdf"}), 0.9)
        ]
        store.get_document_count.return_value = 10
        mock.return_value = store
        yield store


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph"""
    with patch('app.services.knowledge_graph.KnowledgeGraph') as mock:
        graph = Mock()
        graph.get_all_entities.return_value = [
            {"name": "Annual Leave", "type": "benefit"}
        ]
        graph.extract_and_store_entities.return_value = 5
        mock.return_value = graph
        yield graph


@pytest.fixture
def mock_document_processor():
    """Mock document processor"""
    with patch('app.services.document_processor.DocumentProcessor') as mock:
        processor = Mock()
        from langchain.schema import Document

        processor.chunk_document.return_value = [
            Document(
                page_content="Test content",
                metadata={"document_id": "test123", "chunk_id": "chunk1"}
            )
        ]
        processor.extract_metadata.return_value = {
            "file_name": "test.pdf",
            "total_pages": 5
        }
        mock.return_value = processor
        yield processor


def pytest_configure(config):
    """Pytest configuration hook"""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Auto-add markers based on test location/name
    for item in items:
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_settings" in item.nodeid or "test_embeddings" in item.nodeid or "test_chat" in item.nodeid:
            item.add_marker(pytest.mark.unit)
