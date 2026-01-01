from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from app.utils.embeddings import get_embeddings
from config.settings import settings


class VectorStore:
    """Manages vector embeddings and similarity search using ChromaDB"""

    def __init__(self):
        # Set API key in environment based on provider (required for compatibility)
        if settings.llm_provider == "gemini":
            if settings.gemini_api_key:
                os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
        else:
            if settings.openai_api_key:
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        # Initialize embeddings using factory (auto-selects provider)
        self.embeddings = get_embeddings()

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collection name for policy documents
        self.collection_name = "policy_documents"

        # Initialize LangChain Chroma wrapper
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs
        """
        # Add documents to vector store
        ids = self.vector_store.add_documents(documents)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters

        Returns:
            List of relevant Document objects
        """
        if filter_dict:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Dict[str, Any] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters

        Returns:
            List of tuples (Document, score)
        """
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)

        return results

    def delete_documents_by_source(self, file_path: str) -> bool:
        """
        Delete all documents from a specific source file

        Args:
            file_path: Path to the source file

        Returns:
            True if successful
        """
        try:
            # Get collection
            collection = self.chroma_client.get_collection(self.collection_name)

            # Delete documents matching the file_path
            collection.delete(
                where={"file_path": file_path}
            )
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def get_document_count(self) -> int:
        """Get total number of documents in the vector store"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def reset_store(self) -> bool:
        """Reset the entire vector store (use with caution)"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            # Reinitialize
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            return True
        except Exception as e:
            print(f"Error resetting store: {e}")
            return False
