from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # LLM Provider API Keys (auto-detect based on which key is present)
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # ChromaDB (also stores knowledge graph)
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        env="CHROMA_PERSIST_DIRECTORY"
    )

    # Application
    app_name: str = Field(default="Enterprise Policy Chatbot", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")

    # Document Processing
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Embedding Model
    embedding_model: str = Field(default="text-embedding-ada-002")

    # RAG Strategy Configuration
    embedding_strategy: str = Field(
        default="provider",
        env="EMBEDDING_STRATEGY",
        description="Embedding strategy: 'provider' (OpenAI/Gemini) or 'local' (SentenceTransformer)"
    )
    chunking_strategy: str = Field(
        default="fixed",
        env="CHUNKING_STRATEGY",
        description="Chunking strategy: 'fixed' (legacy) or 'section' (header-based)"
    )
    use_mmr_retrieval: bool = Field(
        default=False,
        env="USE_MMR_RETRIEVAL",
        description="Enable MMR-based sentence selection"
    )
    mmr_k: int = Field(
        default=6,
        env="MMR_K",
        description="Number of sentences to select with MMR"
    )
    mmr_lambda: float = Field(
        default=0.7,
        env="MMR_LAMBDA",
        description="MMR lambda parameter (0-1): relevance vs diversity tradeoff"
    )
    n_clusters: int = Field(
        default=6,
        env="N_CLUSTERS",
        description="Number of clusters for section clustering"
    )
    use_llm_refinement: bool = Field(
        default=False,
        env="USE_LLM_REFINEMENT",
        description="Use LLM to refine MMR-selected sentences (increases cost)"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def llm_provider(self) -> str:
        """Auto-detect LLM provider based on available API keys.
        Priority: Gemini > OpenAI"""
        if self.gemini_api_key:
            return "gemini"
        return "openai"

    @model_validator(mode='after')
    def validate_api_keys(self) -> 'Settings':
        """Ensure at least one API key is configured"""
        if not self.gemini_api_key and not self.openai_api_key:
            raise ValueError("Either GEMINI_API_KEY or OPENAI_API_KEY must be set in .env file")
        return self


settings = Settings()
