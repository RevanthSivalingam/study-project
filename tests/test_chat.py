"""
Test chat LLM factory and provider implementations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import LLMResult, Generation
from app.utils.chat import (
    SimpleChatOpenAI,
    SimpleGeminiChat,
    get_chat_llm
)


class TestChatLLMFactory:
    """Test chat LLM factory function"""

    @patch('app.utils.chat.settings')
    def test_get_chat_llm_returns_gemini(self, mock_settings):
        """Factory should return Gemini chat when provider is gemini"""
        mock_settings.llm_provider = "gemini"
        mock_settings.gemini_api_key = "test_key"

        with patch('app.utils.chat.SimpleGeminiChat') as mock_gemini:
            llm = get_chat_llm()
            mock_gemini.assert_called_once()

    @patch('app.utils.chat.settings')
    def test_get_chat_llm_returns_openai(self, mock_settings):
        """Factory should return OpenAI chat when provider is openai"""
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "test_key"

        with patch('app.utils.chat.SimpleChatOpenAI') as mock_openai:
            llm = get_chat_llm()
            mock_openai.assert_called_once()

    @patch('app.utils.chat.settings')
    def test_custom_model_parameter(self, mock_settings):
        """Factory should pass custom model to provider"""
        mock_settings.llm_provider = "gemini"
        mock_settings.gemini_api_key = "test_key"

        with patch('app.utils.chat.SimpleGeminiChat') as mock_gemini:
            llm = get_chat_llm(model="custom-model", temperature=0.5)
            mock_gemini.assert_called_once_with(model="custom-model", temperature=0.5)

    @patch('app.utils.chat.settings')
    def test_default_models(self, mock_settings):
        """Factory should use default models when not specified"""
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "test_key"

        with patch('app.utils.chat.SimpleChatOpenAI') as mock_openai:
            llm = get_chat_llm()
            mock_openai.assert_called_once_with(model="gpt-4", temperature=0)


class TestOpenAIChat:
    """Test OpenAI chat implementation"""

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_initialization(self, mock_settings, mock_openai_client):
        """Should initialize OpenAI client correctly"""
        mock_settings.openai_api_key = "test_key"

        chat = SimpleChatOpenAI(model="gpt-4", temperature=0)

        mock_openai_client.assert_called_once_with(api_key="test_key")
        assert chat.model == "gpt-4"
        assert chat.temperature == 0

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_call_method(self, mock_settings, mock_openai_client):
        """Should call OpenAI API correctly"""
        mock_settings.openai_api_key = "test_key"

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        chat = SimpleChatOpenAI()
        result = chat._call("Test prompt")

        assert result == "Test response"

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_generate_method(self, mock_settings, mock_openai_client):
        """Should generate responses for multiple prompts"""
        mock_settings.openai_api_key = "test_key"

        # Mock responses
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        chat = SimpleChatOpenAI()
        result = chat._generate(["prompt1", "prompt2"])

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 2

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_predict_method(self, mock_settings, mock_openai_client):
        """Predict should call _call method"""
        mock_settings.openai_api_key = "test_key"

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Predicted"))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        chat = SimpleChatOpenAI()
        result = chat.predict("Test")

        assert result == "Predicted"

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_llm_type_property(self, mock_settings, mock_openai_client):
        """Should return correct LLM type"""
        mock_settings.openai_api_key = "test_key"

        chat = SimpleChatOpenAI()

        assert chat._llm_type == "openai-chat"

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_stop_sequences(self, mock_settings, mock_openai_client):
        """Should pass stop sequences to API"""
        mock_settings.openai_api_key = "test_key"

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        chat = SimpleChatOpenAI()
        chat._call("prompt", stop=["STOP", "END"])

        call_args = mock_openai_client.return_value.chat.completions.create.call_args
        assert call_args[1]["stop"] == ["STOP", "END"]


class TestGeminiChat:
    """Test Gemini chat implementation"""

    @patch('app.utils.chat.settings')
    def test_initialization(self, mock_settings):
        """Should initialize Gemini with correct model"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                chat = SimpleGeminiChat(model="gemini-2.0-flash-exp", temperature=0)

                mock_configure.assert_called_once_with(api_key="test_key")
                mock_model.assert_called_once_with("gemini-2.0-flash-exp")
                assert chat.model_name == "gemini-2.0-flash-exp"
                assert chat.temperature == 0

    @patch('app.utils.chat.settings')
    def test_call_method(self, mock_settings):
        """Should call Gemini API correctly"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            mock_client = MagicMock()
            mock_response = Mock()
            mock_response.text = "Gemini response"
            mock_client.generate_content.return_value = mock_response

            with patch('google.generativeai.GenerativeModel', return_value=mock_client):
                chat = SimpleGeminiChat()
                result = chat._call("Test prompt")

                assert result == "Gemini response"
                mock_client.generate_content.assert_called_once()

    @patch('app.utils.chat.settings')
    def test_generate_method(self, mock_settings):
        """Should generate responses for multiple prompts"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            mock_client = MagicMock()
            mock_response = Mock()
            mock_response.text = "Response"
            mock_client.generate_content.return_value = mock_response

            with patch('google.generativeai.GenerativeModel', return_value=mock_client):
                chat = SimpleGeminiChat()
                result = chat._generate(["prompt1", "prompt2"])

                assert isinstance(result, LLMResult)
                assert len(result.generations) == 2

    @patch('app.utils.chat.settings')
    def test_temperature_config(self, mock_settings):
        """Should pass temperature to generation config"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            mock_client = MagicMock()
            mock_response = Mock()
            mock_response.text = "Response"
            mock_client.generate_content.return_value = mock_response

            with patch('google.generativeai.GenerativeModel', return_value=mock_client):
                chat = SimpleGeminiChat(temperature=0.7)
                chat._call("Test")

                call_args = mock_client.generate_content.call_args
                gen_config = call_args[1]['generation_config']
                assert gen_config['temperature'] == 0.7

    @patch('app.utils.chat.settings')
    def test_stop_sequences(self, mock_settings):
        """Should pass stop sequences to generation config"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            mock_client = MagicMock()
            mock_response = Mock()
            mock_response.text = "Response"
            mock_client.generate_content.return_value = mock_response

            with patch('google.generativeai.GenerativeModel', return_value=mock_client):
                chat = SimpleGeminiChat()
                chat._call("prompt", stop=["STOP", "END"])

                call_args = mock_client.generate_content.call_args
                gen_config = call_args[1]['generation_config']
                assert gen_config['stop_sequences'] == ["STOP", "END"]

    @patch('app.utils.chat.settings')
    def test_llm_type_property(self, mock_settings):
        """Should return correct LLM type"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                chat = SimpleGeminiChat()

                assert chat._llm_type == "gemini-chat"


class TestChatCompatibility:
    """Test that both implementations are compatible with LangChain"""

    @patch('app.utils.chat.OpenAI')
    @patch('app.utils.chat.settings')
    def test_openai_langchain_interface(self, mock_settings, mock_openai):
        """OpenAI chat should implement LangChain BaseLLM interface"""
        mock_settings.openai_api_key = "test_key"

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        chat = SimpleChatOpenAI()

        # Should have required LangChain methods
        assert hasattr(chat, '_call')
        assert hasattr(chat, '_generate')
        assert hasattr(chat, '_llm_type')
        assert hasattr(chat, '_identifying_params')

    @patch('app.utils.chat.settings')
    def test_gemini_langchain_interface(self, mock_settings):
        """Gemini chat should implement LangChain BaseLLM interface"""
        mock_settings.gemini_api_key = "test_key"

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                chat = SimpleGeminiChat()

                # Should have required LangChain methods
                assert hasattr(chat, '_call')
                assert hasattr(chat, '_generate')
                assert hasattr(chat, '_llm_type')
                assert hasattr(chat, '_identifying_params')
