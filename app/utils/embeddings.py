"""
Custom embeddings wrapper for OpenAI
Fixes compatibility issues with langchain-openai
"""
from typing import List
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from config.settings import settings


class SimpleOpenAIEmbeddings(Embeddings):
    """Simple OpenAI embeddings wrapper that actually works"""

    def __init__(self):
        # Create OpenAI client without custom httpx - let it use defaults
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "text-embedding-ada-002"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embeddings error: {type(e).__name__}: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding


class SimpleGeminiEmbeddings(Embeddings):
    """Gemini embeddings wrapper for LangChain compatibility"""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            import google.generativeai as genai
            texts = [text.replace("\n", " ") for text in texts]
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            print(f"Gemini embeddings error: {type(e).__name__}: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            import google.generativeai as genai
            text = text.replace("\n", " ")
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Gemini query embedding error: {type(e).__name__}: {str(e)}")
            raise


class SentenceTransformerEmbeddings(Embeddings):
    """Local SentenceTransformer embeddings for privacy and cost efficiency"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            texts = [text.replace("\n", " ") for text in texts]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"SentenceTransformer embeddings error: {type(e).__name__}: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        text = text.replace("\n", " ")
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


def get_embeddings() -> Embeddings:
    """Factory function to get appropriate embeddings based on provider.

    Strategy priority:
    1. If embedding_strategy = "local", use SentenceTransformer
    2. Else use provider-based (Gemini or OpenAI)

    Returns:
        Appropriate Embeddings instance
    """
    # Check if local strategy is requested
    embedding_strategy = getattr(settings, 'embedding_strategy', 'provider')

    if embedding_strategy == "local":
        return SentenceTransformerEmbeddings()
    elif settings.llm_provider == "gemini":
        return SimpleGeminiEmbeddings()
    else:
        # OpenAI fallback (still available and intact)
        return SimpleOpenAIEmbeddings()
