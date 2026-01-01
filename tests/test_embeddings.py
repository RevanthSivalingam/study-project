"""
Test embeddings factory and provider implementations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.utils.embeddings import (
    SimpleOpenAIEmbeddings,
    SimpleGeminiEmbeddings,
    get_embeddings
)


class TestEmbeddingsFactory:
    """Test embeddings factory function"""

    @patch('app.utils.embeddings.settings')
    def test_get_embeddings_returns_gemini(self, mock_settings):
        """Factory should return Gemini embeddings when provider is gemini"""
        mock_settings.llm_provider = "gemini"
        mock_settings.gemini_api_key = "test_key"

        with patch('app.utils.embeddings.SimpleGeminiEmbeddings') as mock_gemini:
            embeddings = get_embeddings()
            mock_gemini.assert_called_once()

    @patch('app.utils.embeddings.settings')
    def test_get_embeddings_returns_openai(self, mock_settings):
        """Factory should return OpenAI embeddings when provider is openai"""
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "test_key"

        with patch('app.utils.embeddings.SimpleOpenAIEmbeddings') as mock_openai:
            embeddings = get_embeddings()
            mock_openai.assert_called_once()


class TestOpenAIEmbeddings:
    """Test OpenAI embeddings implementation"""

    @patch('app.utils.embeddings.OpenAI')
    @patch('app.utils.embeddings.settings')
    def test_initialization(self, mock_settings, mock_openai_client):
        """Should initialize OpenAI client correctly"""
        mock_settings.openai_api_key = "test_key"

        embeddings = SimpleOpenAIEmbeddings()

        mock_openai_client.assert_called_once_with(api_key="test_key")
        assert embeddings.model == "text-embedding-ada-002"

    @patch('app.utils.embeddings.OpenAI')
    @patch('app.utils.embeddings.settings')
    def test_embed_documents(self, mock_settings, mock_openai_client):
        """Should embed multiple documents"""
        mock_settings.openai_api_key = "test_key"

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        embeddings = SimpleOpenAIEmbeddings()
        result = embeddings.embed_documents(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch('app.utils.embeddings.OpenAI')
    @patch('app.utils.embeddings.settings')
    def test_embed_query(self, mock_settings, mock_openai_client):
        """Should embed a single query"""
        mock_settings.openai_api_key = "test_key"

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.7, 0.8, 0.9])]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        embeddings = SimpleOpenAIEmbeddings()
        result = embeddings.embed_query("test query")

        assert result == [0.7, 0.8, 0.9]

    @patch('app.utils.embeddings.OpenAI')
    @patch('app.utils.embeddings.settings')
    def test_newline_replacement(self, mock_settings, mock_openai_client):
        """Should replace newlines with spaces"""
        mock_settings.openai_api_key = "test_key"
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        embeddings = SimpleOpenAIEmbeddings()
        embeddings.embed_query("text\nwith\nnewlines")

        # Verify newlines were replaced
        call_args = mock_openai_client.return_value.embeddings.create.call_args
        assert "\n" not in call_args[1]["input"][0]


class TestGeminiEmbeddings:
    """Test Gemini embeddings implementation"""

    @patch('app.utils.embeddings.settings')
    def test_initialization(self, mock_settings):
        """Should initialize Gemini with correct model"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure') as mock_configure:
            embeddings = SimpleGeminiEmbeddings()

            mock_configure.assert_called_once_with(api_key="test_key")
            assert embeddings.model == "models/text-embedding-004"

    @patch('app.utils.embeddings.settings')
    def test_embed_documents(self, mock_settings):
        """Should embed multiple documents with correct task_type"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.embed_content') as mock_embed:
                mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}

                embeddings = SimpleGeminiEmbeddings()
                result = embeddings.embed_documents(["text1", "text2"])

                assert len(result) == 2
                # Verify task_type is "retrieval_document" for documents
                calls = mock_embed.call_args_list
                for call in calls:
                    assert call[1]['task_type'] == "retrieval_document"

    @patch('app.utils.embeddings.settings')
    def test_embed_query(self, mock_settings):
        """Should embed query with correct task_type"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.embed_content') as mock_embed:
                mock_embed.return_value = {'embedding': [0.4, 0.5, 0.6]}

                embeddings = SimpleGeminiEmbeddings()
                result = embeddings.embed_query("test query")

                assert result == [0.4, 0.5, 0.6]
                # Verify task_type is "retrieval_query" for queries
                assert mock_embed.call_args[1]['task_type'] == "retrieval_query"

    @patch('app.utils.embeddings.settings')
    def test_error_handling(self, mock_settings):
        """Should handle and re-raise errors properly"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.embed_content', side_effect=Exception("API Error")):
                embeddings = SimpleGeminiEmbeddings()

                with pytest.raises(Exception, match="API Error"):
                    embeddings.embed_query("test")


class TestEmbeddingDimensions:
    """Test that embeddings return correct dimensions"""

    @patch('app.utils.embeddings.OpenAI')
    @patch('app.utils.embeddings.settings')
    def test_openai_dimension(self, mock_settings, mock_openai):
        """OpenAI embeddings should be 1536 dimensions"""
        mock_settings.openai_api_key = "test_key"

        # Mock 1536-dim embedding
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        embeddings = SimpleOpenAIEmbeddings()
        result = embeddings.embed_query("test")

        assert len(result) == 1536

    @patch('app.utils.embeddings.settings')
    def test_gemini_dimension(self, mock_settings):
        """Gemini embeddings should be 768 dimensions"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.embed_content') as mock_embed:
                # Mock 768-dim embedding
                mock_embed.return_value = {'embedding': [0.1] * 768}

                embeddings = SimpleGeminiEmbeddings()
                result = embeddings.embed_query("test")

                assert len(result) == 768
