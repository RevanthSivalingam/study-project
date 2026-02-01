"""
RAG evaluation metrics for measuring retrieval quality.

Implements standard information retrieval metrics:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Confidence scoring
"""
from typing import List, Set


class RAGEvaluator:
    """
    Evaluates RAG retrieval quality using standard IR metrics.
    """

    def calculate_precision_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@k.

        Precision@k = (# relevant items in top k) / k

        Args:
            retrieved: List of retrieved item indices (in rank order)
            relevant: List of relevant item indices
            k: Cutoff rank

        Returns:
            Precision@k score (0-1)
        """
        if k <= 0 or not retrieved:
            return 0.0

        # Consider only top k retrieved items
        top_k = retrieved[:k]
        relevant_set = set(relevant)

        # Count how many of top k are relevant
        num_relevant = sum(1 for item in top_k if item in relevant_set)

        return num_relevant / k

    def calculate_recall_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@k.

        Recall@k = (# relevant items in top k) / (# total relevant items)

        Args:
            retrieved: List of retrieved item indices (in rank order)
            relevant: List of relevant item indices
            k: Cutoff rank

        Returns:
            Recall@k score (0-1)
        """
        if not relevant or k <= 0 or not retrieved:
            return 0.0

        # Consider only top k retrieved items
        top_k = retrieved[:k]
        relevant_set = set(relevant)

        # Count how many relevant items were retrieved
        num_relevant_retrieved = sum(1 for item in top_k if item in relevant_set)

        return num_relevant_retrieved / len(relevant)

    def calculate_mrr(
        self,
        retrieved: List[int],
        relevant: List[int]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / (rank of first relevant item)

        Args:
            retrieved: List of retrieved item indices (in rank order)
            relevant: List of relevant item indices

        Returns:
            MRR score (0-1)
        """
        if not retrieved or not relevant:
            return 0.0

        relevant_set = set(relevant)

        # Find rank of first relevant item (1-indexed)
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant_set:
                return 1.0 / rank

        # No relevant items found
        return 0.0

    def calculate_f1_score(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate F1 score at k.

        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            retrieved: List of retrieved item indices
            relevant: List of relevant item indices
            k: Cutoff rank

        Returns:
            F1 score (0-1)
        """
        precision = self.calculate_precision_at_k(retrieved, relevant, k)
        recall = self.calculate_recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_confidence(
        self,
        num_retrieved: int,
        k: int = 5,
        baseline_threshold: float = 0.5
    ) -> float:
        """
        Calculate confidence score based on retrieval results.

        Confidence increases with:
        - More sentences retrieved (up to k)
        - Meeting minimum threshold

        Args:
            num_retrieved: Number of sentences/items retrieved
            k: Target number of items (default 5)
            baseline_threshold: Minimum confidence threshold

        Returns:
            Confidence score (0-1)
        """
        if num_retrieved <= 0:
            return 0.0

        # Base confidence from retrieval count
        retrieval_confidence = min(num_retrieved, k) / k

        # Ensure minimum threshold
        confidence = max(baseline_threshold, retrieval_confidence)

        return min(1.0, confidence)

    def calculate_ndcg(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).

        NDCG measures ranking quality with position-based discounting.

        Args:
            retrieved: List of retrieved item indices (in rank order)
            relevant: List of relevant item indices
            k: Cutoff rank

        Returns:
            NDCG@k score (0-1)
        """
        import math

        if not relevant or not retrieved or k <= 0:
            return 0.0

        relevant_set = set(relevant)
        top_k = retrieved[:k]

        # Calculate DCG
        dcg = 0.0
        for rank, item in enumerate(top_k, start=1):
            if item in relevant_set:
                # Binary relevance (1 if relevant, 0 otherwise)
                dcg += 1.0 / math.log2(rank + 1)

        # Calculate ideal DCG (all relevant items ranked first)
        num_relevant_in_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant_in_k + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_retrieval(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int = 5
    ) -> dict:
        """
        Perform comprehensive evaluation of retrieval results.

        Args:
            retrieved: List of retrieved item indices
            relevant: List of relevant item indices
            k: Cutoff rank for metrics

        Returns:
            Dictionary with all evaluation metrics
        """
        return {
            'precision_at_k': self.calculate_precision_at_k(retrieved, relevant, k),
            'recall_at_k': self.calculate_recall_at_k(retrieved, relevant, k),
            'f1_at_k': self.calculate_f1_score(retrieved, relevant, k),
            'mrr': self.calculate_mrr(retrieved, relevant),
            'ndcg_at_k': self.calculate_ndcg(retrieved, relevant, k),
            'num_retrieved': len(retrieved),
            'num_relevant': len(relevant),
            'confidence': self.calculate_confidence(len(retrieved), k)
        }

    def compare_retrievals(
        self,
        retrieval_a: List[int],
        retrieval_b: List[int],
        relevant: List[int],
        k: int = 5
    ) -> dict:
        """
        Compare two retrieval methods.

        Args:
            retrieval_a: First retrieval result
            retrieval_b: Second retrieval result
            relevant: Ground truth relevant items
            k: Cutoff rank

        Returns:
            Dictionary comparing both methods
        """
        metrics_a = self.evaluate_retrieval(retrieval_a, relevant, k)
        metrics_b = self.evaluate_retrieval(retrieval_b, relevant, k)

        return {
            'method_a': metrics_a,
            'method_b': metrics_b,
            'improvements': {
                'precision': metrics_b['precision_at_k'] - metrics_a['precision_at_k'],
                'recall': metrics_b['recall_at_k'] - metrics_a['recall_at_k'],
                'f1': metrics_b['f1_at_k'] - metrics_a['f1_at_k'],
                'mrr': metrics_b['mrr'] - metrics_a['mrr'],
                'ndcg': metrics_b['ndcg_at_k'] - metrics_a['ndcg_at_k']
            }
        }
