"""
Section clustering using K-Means for semantic grouping.

Clusters document sections to enable efficient semantic fallback retrieval
when knowledge graph queries don't yield results.
"""
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from app.services.section_processor import Section


class SectionClusterer:
    """
    Clusters document sections using K-Means algorithm.

    Enables semantic fallback by grouping similar sections and selecting
    the most relevant cluster for a query.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        persist_directory: str = "./data/chroma_db"
    ):
        """
        Initialize the SectionClusterer.

        Args:
            n_clusters: Number of clusters to create
            persist_directory: Directory to save/load cluster models
        """
        self.n_clusters = n_clusters
        self.persist_directory = persist_directory
        self.models_dir = os.path.join(persist_directory, "cluster_models")
        os.makedirs(self.models_dir, exist_ok=True)

        # K-Means model
        self.kmeans: Optional[KMeans] = None
        self.cluster_centroids: Optional[np.ndarray] = None

        # Section→cluster mapping
        self.section_clusters: Dict[str, int] = {}  # section_title -> cluster_id
        self.cluster_sections: Dict[int, List[str]] = {}  # cluster_id -> [section_titles]

        self.is_fitted = False

    def fit(self, sections: List[Section], embeddings: List[List[float]]) -> None:
        """
        Fit K-Means clustering on section embeddings.

        Args:
            sections: List of document sections
            embeddings: List of embedding vectors for each section
        """
        if len(sections) != len(embeddings):
            raise ValueError("Number of sections must match number of embeddings")

        if len(sections) < self.n_clusters:
            print(f"Warning: Only {len(sections)} sections, using {len(sections)} clusters")
            self.n_clusters = max(1, len(sections))

        # Convert embeddings to numpy array
        X = np.array(embeddings)

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X)
        self.cluster_centroids = self.kmeans.cluster_centers_

        # Build mappings
        self.section_clusters = {}
        self.cluster_sections = {i: [] for i in range(self.n_clusters)}

        for section, cluster_id in zip(sections, cluster_labels):
            self.section_clusters[section.title] = int(cluster_id)
            self.cluster_sections[int(cluster_id)].append(section.title)

        self.is_fitted = True

        print(f"Clustered {len(sections)} sections into {self.n_clusters} clusters")
        for cluster_id, section_titles in self.cluster_sections.items():
            print(f"  Cluster {cluster_id}: {len(section_titles)} sections")

    def select_cluster(self, query_embedding: List[float]) -> int:
        """
        Select the most relevant cluster for a query based on centroid similarity.

        Args:
            query_embedding: Embedding vector for the query

        Returns:
            Cluster ID of the most similar cluster
        """
        if not self.is_fitted or self.cluster_centroids is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")

        query_vec = np.array(query_embedding)

        # Calculate cosine similarity with all centroids
        similarities = []
        for centroid in self.cluster_centroids:
            sim = self._cosine_similarity(query_vec, centroid)
            similarities.append(sim)

        # Return cluster with highest similarity
        best_cluster = int(np.argmax(similarities))

        return best_cluster

    def get_cluster_sections(self, cluster_id: int) -> List[str]:
        """
        Get all section titles in a cluster.

        Args:
            cluster_id: The cluster ID

        Returns:
            List of section titles in the cluster
        """
        return self.cluster_sections.get(cluster_id, [])

    def get_section_cluster(self, section_title: str) -> Optional[int]:
        """
        Get the cluster ID for a section.

        Args:
            section_title: Title of the section

        Returns:
            Cluster ID or None if section not found
        """
        return self.section_clusters.get(section_title)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of sections across clusters.

        Returns:
            Dictionary mapping cluster_id -> section_count
        """
        return {
            cluster_id: len(section_titles)
            for cluster_id, section_titles in self.cluster_sections.items()
        }

    def save(self) -> None:
        """Save cluster model to disk."""
        if not self.is_fitted:
            print("Warning: No fitted model to save")
            return

        model_path = os.path.join(self.models_dir, "section_clusterer.pkl")

        save_data = {
            'kmeans': self.kmeans,
            'cluster_centroids': self.cluster_centroids,
            'section_clusters': self.section_clusters,
            'cluster_sections': self.cluster_sections,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Section clusterer saved to {model_path}")

    def load(self) -> bool:
        """
        Load cluster model from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        model_path = os.path.join(self.models_dir, "section_clusterer.pkl")

        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)

            self.kmeans = save_data['kmeans']
            self.cluster_centroids = save_data['cluster_centroids']
            self.section_clusters = save_data['section_clusters']
            self.cluster_sections = save_data['cluster_sections']
            self.n_clusters = save_data['n_clusters']
            self.is_fitted = save_data['is_fitted']

            print(f"Section clusterer loaded from {model_path}")
            print(f"  - Clusters: {self.n_clusters}")
            print(f"  - Sections: {len(self.section_clusters)}")

            return True
        except Exception as e:
            print(f"Error loading section clusterer: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get statistics about clustering.

        Returns:
            Dictionary with clustering statistics
        """
        if not self.is_fitted:
            return {'is_fitted': False}

        distribution = self.get_cluster_distribution()

        return {
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters,
            'total_sections': len(self.section_clusters),
            'cluster_distribution': distribution,
            'min_cluster_size': min(distribution.values()) if distribution else 0,
            'max_cluster_size': max(distribution.values()) if distribution else 0,
            'avg_cluster_size': sum(distribution.values()) / len(distribution) if distribution else 0
        }
