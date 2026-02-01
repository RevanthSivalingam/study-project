"""
TF-IDF-based term learning for query normalization and key term extraction.

This module learns generic (low-information) terms and key terms from the corpus
to enable intelligent query processing and knowledge graph construction.
"""
import os
import pickle
from typing import List, Set, Dict, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from app.services.section_processor import Section


class TermLearner:
    """
    Learns generic and key terms from document corpus using TF-IDF.

    Generic Terms: Low-information words to filter from queries (bottom 15% TF-IDF)
    Key Terms: High-information unigrams/bigrams for KG construction (top 40 terms)
    """

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize the TermLearner.

        Args:
            persist_directory: Directory to save/load learned models
        """
        self.persist_directory = persist_directory
        self.models_dir = os.path.join(persist_directory, "term_models")
        os.makedirs(self.models_dir, exist_ok=True)

        # TF-IDF vectorizer for learning
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000,   # Limit vocabulary size
            min_df=2,            # Term must appear in at least 2 documents
            lowercase=True
        )

        # Learned terms
        self.generic_terms: Set[str] = set()
        self.key_terms: List[str] = []
        self.term_scores: Dict[str, float] = {}
        self.is_fitted = False

    def learn_generic_terms(self, sections: List[Section]) -> Set[str]:
        """
        Learn generic (low-information) terms from corpus.

        Identifies terms in the bottom 15% of TF-IDF scores as generic.

        Args:
            sections: List of document sections

        Returns:
            Set of generic terms
        """
        if not sections:
            return set()

        # Extract text from sections
        texts = [section.content for section in sections]

        # Fit TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        except ValueError as e:
            print(f"TF-IDF fitting error: {e}")
            return set()

        # Calculate mean TF-IDF score for each term
        feature_names = self.vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Store scores
        self.term_scores = dict(zip(feature_names, mean_scores))

        # Identify bottom 15% as generic terms
        sorted_terms = sorted(self.term_scores.items(), key=lambda x: x[1])
        cutoff_idx = max(1, int(len(sorted_terms) * 0.15))
        self.generic_terms = {term for term, _ in sorted_terms[:cutoff_idx]}

        print(f"Learned {len(self.generic_terms)} generic terms (bottom 15% TF-IDF)")

        return self.generic_terms

    def learn_key_terms(self, sections: List[Section], top_n: int = 40) -> List[str]:
        """
        Learn key terms (high-information unigrams/bigrams) from corpus.

        Identifies top N terms by TF-IDF score.

        Args:
            sections: List of document sections
            top_n: Number of key terms to extract

        Returns:
            List of key terms sorted by importance
        """
        if not self.is_fitted:
            # Ensure TF-IDF is fitted
            self.learn_generic_terms(sections)

        if not self.term_scores:
            return []

        # Sort terms by TF-IDF score (descending)
        sorted_terms = sorted(
            self.term_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top N terms
        self.key_terms = [term for term, _ in sorted_terms[:top_n]]

        print(f"Learned {len(self.key_terms)} key terms (top {top_n} by TF-IDF)")

        return self.key_terms

    def normalize_query(self, query: str) -> str:
        """
        Normalize a query by removing generic terms.

        Args:
            query: The query string to normalize

        Returns:
            Normalized query string
        """
        if not self.generic_terms:
            return query

        # Tokenize and filter
        words = query.lower().split()
        filtered_words = [
            word for word in words
            if word not in self.generic_terms
        ]

        # Also remove single-character words and numbers
        filtered_words = [
            word for word in filtered_words
            if len(word) > 1 and not word.isdigit()
        ]

        normalized = ' '.join(filtered_words)

        # If normalization removed everything, return original
        return normalized if normalized.strip() else query

    def extract_key_terms_from_text(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key terms from a specific text using the learned vocabulary.

        Args:
            text: Text to extract terms from
            top_n: Number of top terms to return

        Returns:
            List of key terms found in the text
        """
        if not self.is_fitted:
            return []

        # Transform text using learned vectorizer
        try:
            tfidf_vector = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()

            # Get scores for this text
            scores = tfidf_vector.toarray()[0]

            # Get top terms
            top_indices = scores.argsort()[-top_n:][::-1]
            top_terms = [
                feature_names[idx]
                for idx in top_indices
                if scores[idx] > 0
            ]

            return top_terms
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []

    def get_term_frequencies(self, sections: List[Section]) -> Dict[str, int]:
        """
        Get term frequency counts across all sections.

        Args:
            sections: List of document sections

        Returns:
            Dictionary mapping terms to their frequencies
        """
        all_text = ' '.join(section.content.lower() for section in sections)
        words = all_text.split()

        # Filter stop words and short words
        words = [w for w in words if len(w) > 2 and w.isalpha()]

        return dict(Counter(words))

    def save(self) -> None:
        """Save learned models to disk."""
        if not self.is_fitted:
            print("Warning: No fitted model to save")
            return

        model_path = os.path.join(self.models_dir, "term_learner.pkl")

        save_data = {
            'vectorizer': self.vectorizer,
            'generic_terms': self.generic_terms,
            'key_terms': self.key_terms,
            'term_scores': self.term_scores,
            'is_fitted': self.is_fitted
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Term learner saved to {model_path}")

    def load(self) -> bool:
        """
        Load learned models from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        model_path = os.path.join(self.models_dir, "term_learner.pkl")

        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            self.vectorizer = save_data['vectorizer']
            self.generic_terms = save_data['generic_terms']
            self.key_terms = save_data['key_terms']
            self.term_scores = save_data['term_scores']
            self.is_fitted = save_data['is_fitted']

            print(f"Term learner loaded from {model_path}")
            print(f"  - Generic terms: {len(self.generic_terms)}")
            print(f"  - Key terms: {len(self.key_terms)}")

            return True
        except Exception as e:
            print(f"Error loading term learner: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get statistics about learned terms.

        Returns:
            Dictionary with term learning statistics
        """
        return {
            'is_fitted': self.is_fitted,
            'num_generic_terms': len(self.generic_terms),
            'num_key_terms': len(self.key_terms),
            'vocabulary_size': len(self.term_scores),
            'generic_terms_sample': list(self.generic_terms)[:10],
            'key_terms_sample': self.key_terms[:10]
        }
