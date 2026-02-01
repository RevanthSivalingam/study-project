"""
Integration tests for RAG strategy switching
"""
import pytest
from unittest.mock import Mock, patch
from app.services.rag_service import RAGService
from config.settings import settings


@pytest.fixture
def mock_enhanced_service():
    """Mock enhanced RAG service"""
    mock = Mock()
    mock.process_document.return_value = {
        "success": True,
        "sections_created": 10,
        "key_terms_learned": 40
    }
    mock.query.return_value = Mock(
        answer="Test answer",
        sources=[],
        confidence_score=0.9,
        retrieval_method="kg_guided"
    )
    mock.get_stats.return_value = {
        "total_sections": 10,
        "strategy": "enhanced"
    }
    return mock


def test_legacy_strategy_initialization():
    """Test that legacy strategy initializes correctly"""
    with patch.object(settings, 'chunking_strategy', 'fixed'):
        with patch.object(settings, 'use_mmr_retrieval', False):
            service = RAGService()
            assert service.is_enhanced is False
            assert hasattr(service, 'document_processor')
            assert hasattr(service, 'vector_store')


def test_enhanced_strategy_initialization():
    """Test that enhanced strategy initializes correctly"""
    with patch.object(settings, 'chunking_strategy', 'section'):
        with patch.object(settings, 'use_mmr_retrieval', True):
            service = RAGService()
            assert service.is_enhanced is True
            assert hasattr(service, 'strategy')


def test_process_document_delegation_enhanced(mock_enhanced_service):
    """Test document processing delegates to enhanced strategy"""
    with patch.object(settings, 'chunking_strategy', 'section'):
        with patch.object(settings, 'use_mmr_retrieval', True):
            with patch('app.services.rag_service.EnhancedRAGService', return_value=mock_enhanced_service):
                service = RAGService()
                result = service.process_document("test.pdf", "policy")

                assert result["success"] is True
                assert "sections_created" in result
                mock_enhanced_service.process_document.assert_called_once()


def test_query_delegation_enhanced(mock_enhanced_service):
    """Test query delegates to enhanced strategy"""
    with patch.object(settings, 'chunking_strategy', 'section'):
        with patch.object(settings, 'use_mmr_retrieval', True):
            with patch('app.services.rag_service.EnhancedRAGService', return_value=mock_enhanced_service):
                service = RAGService()
                response = service.query("test question")

                assert response.answer == "Test answer"
                assert response.retrieval_method == "kg_guided"
                mock_enhanced_service.query.assert_called_once()


def test_get_stats_delegation_enhanced(mock_enhanced_service):
    """Test stats delegates to enhanced strategy"""
    with patch.object(settings, 'chunking_strategy', 'section'):
        with patch.object(settings, 'use_mmr_retrieval', True):
            with patch('app.services.rag_service.EnhancedRAGService', return_value=mock_enhanced_service):
                service = RAGService()
                stats = service.get_stats()

                assert "strategy" in stats
                mock_enhanced_service.get_stats.assert_called_once()


def test_backward_compatibility_response_schema():
    """Test that response schema is backward compatible"""
    from app.models.schemas import ChatResponse, SourceReference

    # Old-style response (without enhanced fields)
    response_old = ChatResponse(
        answer="Test answer",
        sources=[],
        confidence_score=0.8
    )

    assert response_old.answer == "Test answer"
    assert response_old.retrieval_method is None  # Optional field

    # New-style response (with enhanced fields)
    response_new = ChatResponse(
        answer="Test answer",
        sources=[],
        confidence_score=0.9,
        retrieval_method="kg_guided",
        mmr_sentences_used=6,
        cluster_id=2
    )

    assert response_new.retrieval_method == "kg_guided"
    assert response_new.mmr_sentences_used == 6


def test_settings_validation():
    """Test that settings are properly configured"""
    # Check that new settings exist
    assert hasattr(settings, 'chunking_strategy')
    assert hasattr(settings, 'embedding_strategy')
    assert hasattr(settings, 'use_mmr_retrieval')
    assert hasattr(settings, 'mmr_k')
    assert hasattr(settings, 'mmr_lambda')
    assert hasattr(settings, 'n_clusters')
    assert hasattr(settings, 'use_llm_refinement')

    # Check default values
    assert settings.chunking_strategy in ['fixed', 'section']
    assert settings.embedding_strategy in ['provider', 'local']
    assert isinstance(settings.mmr_k, int)
    assert 0.0 <= settings.mmr_lambda <= 1.0


@pytest.mark.parametrize("chunking,mmr,expected_enhanced", [
    ('fixed', False, False),
    ('fixed', True, False),
    ('section', False, False),
    ('section', True, True),
])
def test_strategy_selection_matrix(chunking, mmr, expected_enhanced):
    """Test strategy selection based on settings combinations"""
    with patch.object(settings, 'chunking_strategy', chunking):
        with patch.object(settings, 'use_mmr_retrieval', mmr):
            if expected_enhanced:
                # Mock enhanced service to avoid initialization
                with patch('app.services.rag_service.EnhancedRAGService'):
                    service = RAGService()
            else:
                service = RAGService()

            assert service.is_enhanced == expected_enhanced
