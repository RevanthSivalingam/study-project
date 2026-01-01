"""
Test configuration and provider detection
"""
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from config.settings import Settings


class TestProviderDetection:
    """Test LLM provider auto-detection logic"""

    def test_gemini_provider_when_gemini_key_present(self, monkeypatch):
        """Should use Gemini when GEMINI_API_KEY is set"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")

        settings = Settings(_env_file=None)  # Don't load from .env

        assert settings.llm_provider == "gemini"
        assert settings.gemini_api_key == "test_gemini_key"

    def test_openai_provider_when_only_openai_key(self, monkeypatch):
        """Should use OpenAI when only OPENAI_API_KEY is set"""
        # Must set all vars to avoid loading from .env
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")
        # Explicitly set GEMINI_API_KEY to empty to override .env
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(_env_file=None)  # Don't load from .env

        assert settings.llm_provider == "openai"
        assert settings.openai_api_key == "test_openai_key"

    def test_error_when_no_api_keys(self, monkeypatch):
        """Should raise error when neither API key is set"""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)  # Don't load from .env

        assert "Either GEMINI_API_KEY or OPENAI_API_KEY must be set" in str(exc_info.value)

    def test_gemini_priority_over_openai(self, monkeypatch):
        """Gemini should be preferred when both keys are present"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")

        settings = Settings(_env_file=None)  # Don't load from .env

        # Gemini should be chosen even though both exist
        assert settings.llm_provider == "gemini"


class TestSettingsConfiguration:
    """Test other settings configuration"""

    def test_default_values(self, monkeypatch):
        """Test default configuration values"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(_env_file=None)  # Don't load from .env

        assert settings.app_name == "Enterprise Policy Chatbot"
        assert settings.app_version == "1.0.0"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.embedding_model == "text-embedding-ada-002"

    def test_custom_values_from_env(self, monkeypatch):
        """Test custom configuration from environment"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        monkeypatch.setenv("APP_NAME", "Custom Chatbot")
        monkeypatch.setenv("DEBUG", "False")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_data")

        settings = Settings(_env_file=None)  # Don't load from .env

        assert settings.app_name == "Custom Chatbot"
        assert settings.debug is False

    def test_chroma_persist_directory(self, monkeypatch):
        """Test ChromaDB persistence directory configuration"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/custom/path")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(_env_file=None)  # Don't load from .env

        assert settings.chroma_persist_directory == "/custom/path"
