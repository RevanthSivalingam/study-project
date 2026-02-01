"""
Maximal Marginal Relevance (MMR) sentence retrieval.

Implements MMR algorithm to balance relevance and diversity in sentence selection,
reducing redundancy while maintaining comprehensive coverage.
"""
from typing import List, Tuple
import numpy as np
from langchain.embeddings.base import Embeddings


class MMRRetriever:
    """
    Retrieves sentences using Maximal Marginal Relevance (MMR).

    MMR Score = λ * Sim(query, sentence) - (1-λ) * max(Sim(sentence, selected))

    Where:
    - λ controls relevance vs diversity tradeoff (default 0.7 = 70% relevance)
    - Higher λ = more relevant but potentially redundant
    - Lower λ = more diverse but potentially less relevant
    """

    def __init__(self, embeddings: Embeddings):
        """
        Initialize MMR retriever.

        Args:
            embeddings: Embeddings model for computing sentence vectors
        """
        self.embeddings = embeddings

    def select_sentences(
        self,
        query: str,
        sentences: List[str],
        k: int = 6,
        lambda_param: float = 0.7
    ) -> List[str]:
        """
        Select k sentences using MMR algorithm.

        Args:
            query: The search query
            sentences: List of candidate sentences
            k: Number of sentences to select
            lambda_param: Relevance vs diversity tradeoff (0-1)

        Returns:
            List of selected sentences (up to k)
        """
        if not sentences:
            return []

        # Ensure we don't try to select more sentences than available
        k = min(k, len(sentences))

        # Tokenize sentences if they're joined
        if isinstance(sentences, str):
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(sentences)

        # Get embeddings for query and all sentences
        query_embedding = self.embeddings.embed_query(query)
        sentence_embeddings = self.embeddings.embed_documents(sentences)

        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        sentence_vecs = np.array(sentence_embeddings)

        # Calculate relevance scores (similarity to query)
        relevance_scores = [
            self._cosine_similarity(query_vec, sent_vec)
            for sent_vec in sentence_vecs
        ]

        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(sentences)))

        # Select first sentence (most relevant)
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select remaining sentences
        for _ in range(k - 1):
            if not remaining_indices:
                break

            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to already selected)
                max_similarity = max(
                    self._cosine_similarity(
                        sentence_vecs[idx],
                        sentence_vecs[selected_idx]
                    )
                    for selected_idx in selected_indices
                )

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            # Select sentence with highest MMR score
            best_idx_in_remaining = int(np.argmax(mmr_scores))
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected sentences in original order
        selected_indices.sort()
        selected_sentences = [sentences[idx] for idx in selected_indices]

        return selected_sentences

    def select_sentences_with_indices(
        self,
        query: str,
        sentences: List[str],
        k: int = 6,
        lambda_param: float = 0.7
    ) -> Tuple[List[str], List[int]]:
        """
        Select sentences and return both sentences and their indices.

        Args:
            query: The search query
            sentences: List of candidate sentences
            k: Number of sentences to select
            lambda_param: Relevance vs diversity tradeoff (0-1)

        Returns:
            Tuple of (selected_sentences, selected_indices)
        """
        if not sentences:
            return [], []

        k = min(k, len(sentences))

        # Get embeddings
        query_embedding = self.embeddings.embed_query(query)
        sentence_embeddings = self.embeddings.embed_documents(sentences)

        query_vec = np.array(query_embedding)
        sentence_vecs = np.array(sentence_embeddings)

        # Calculate relevance scores
        relevance_scores = [
            self._cosine_similarity(query_vec, sent_vec)
            for sent_vec in sentence_vecs
        ]

        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(sentences)))

        # Select first sentence
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select remaining sentences
        for _ in range(k - 1):
            if not remaining_indices:
                break

            mmr_scores = []

            for idx in remaining_indices:
                relevance = relevance_scores[idx]
                max_similarity = max(
                    self._cosine_similarity(
                        sentence_vecs[idx],
                        sentence_vecs[selected_idx]
                    )
                    for selected_idx in selected_indices
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)

            best_idx_in_remaining = int(np.argmax(mmr_scores))
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return in original order
        sorted_indices = sorted(selected_indices)
        selected_sentences = [sentences[idx] for idx in sorted_indices]

        return selected_sentences, sorted_indices

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def rerank_by_relevance(
        self,
        query: str,
        sentences: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rerank sentences by relevance to query.

        Args:
            query: The search query
            sentences: List of sentences to rank

        Returns:
            List of (sentence, relevance_score) tuples, sorted by score
        """
        if not sentences:
            return []

        # Get embeddings
        query_embedding = self.embeddings.embed_query(query)
        sentence_embeddings = self.embeddings.embed_documents(sentences)

        query_vec = np.array(query_embedding)
        sentence_vecs = np.array(sentence_embeddings)

        # Calculate relevance scores
        scored_sentences = []
        for sentence, sent_vec in zip(sentences, sentence_vecs):
            score = self._cosine_similarity(query_vec, sent_vec)
            scored_sentences.append((sentence, score))

        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        return scored_sentences


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK.

    Args:
        text: Text to tokenize

    Returns:
        List of sentences
    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
