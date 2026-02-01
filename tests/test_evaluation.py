"""
Tests for RAGEvaluator
"""
import pytest
from app.services.evaluation import RAGEvaluator


@pytest.fixture
def evaluator():
    return RAGEvaluator()


def test_precision_at_k_perfect(evaluator):
    """Test precision with perfect retrieval"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 1, 2, 3, 4]

    precision = evaluator.calculate_precision_at_k(retrieved, relevant, k=5)
    assert precision == 1.0


def test_precision_at_k_partial(evaluator):
    """Test precision with partial matches"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 2, 4]

    precision = evaluator.calculate_precision_at_k(retrieved, relevant, k=5)
    # 3 relevant out of 5 retrieved
    assert precision == 0.6


def test_precision_at_k_no_relevant(evaluator):
    """Test precision when no relevant items retrieved"""
    retrieved = [0, 1, 2]
    relevant = [5, 6, 7]

    precision = evaluator.calculate_precision_at_k(retrieved, relevant, k=3)
    assert precision == 0.0


def test_recall_at_k_perfect(evaluator):
    """Test recall with perfect retrieval"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 1, 2]

    recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=5)
    assert recall == 1.0


def test_recall_at_k_partial(evaluator):
    """Test recall with partial retrieval"""
    retrieved = [0, 1, 2]
    relevant = [0, 2, 4, 6]

    recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=3)
    # 2 relevant found out of 4 total relevant
    assert recall == 0.5


def test_recall_at_k_no_relevant(evaluator):
    """Test recall when no relevant items exist"""
    retrieved = [0, 1, 2]
    relevant = []

    recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=3)
    assert recall == 0.0


def test_mrr_first_position(evaluator):
    """Test MRR when first result is relevant"""
    retrieved = [5, 1, 2, 3]
    relevant = [5]

    mrr = evaluator.calculate_mrr(retrieved, relevant)
    assert mrr == 1.0


def test_mrr_second_position(evaluator):
    """Test MRR when relevant item is second"""
    retrieved = [0, 5, 2, 3]
    relevant = [5]

    mrr = evaluator.calculate_mrr(retrieved, relevant)
    assert mrr == 0.5


def test_mrr_no_relevant(evaluator):
    """Test MRR when no relevant items found"""
    retrieved = [0, 1, 2]
    relevant = [5, 6]

    mrr = evaluator.calculate_mrr(retrieved, relevant)
    assert mrr == 0.0


def test_f1_score(evaluator):
    """Test F1 score calculation"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 2, 4, 6, 8]

    f1 = evaluator.calculate_f1_score(retrieved, relevant, k=5)

    # Precision = 3/5, Recall = 3/5
    # F1 = 2 * (0.6 * 0.6) / (0.6 + 0.6) = 0.6
    assert abs(f1 - 0.6) < 0.001


def test_f1_score_zero(evaluator):
    """Test F1 score when precision and recall are zero"""
    retrieved = [0, 1, 2]
    relevant = [5, 6, 7]

    f1 = evaluator.calculate_f1_score(retrieved, relevant, k=3)
    assert f1 == 0.0


def test_calculate_confidence_perfect(evaluator):
    """Test confidence with perfect retrieval"""
    confidence = evaluator.calculate_confidence(num_retrieved=5, k=5)
    assert confidence == 1.0


def test_calculate_confidence_partial(evaluator):
    """Test confidence with partial retrieval"""
    confidence = evaluator.calculate_confidence(num_retrieved=3, k=5)
    # 3/5 = 0.6, but should be at least baseline (0.5)
    assert confidence == 0.6


def test_calculate_confidence_below_baseline(evaluator):
    """Test confidence below baseline threshold"""
    confidence = evaluator.calculate_confidence(num_retrieved=1, k=5, baseline_threshold=0.5)
    # 1/5 = 0.2, but baseline is 0.5
    assert confidence == 0.5


def test_calculate_confidence_zero(evaluator):
    """Test confidence with no results"""
    confidence = evaluator.calculate_confidence(num_retrieved=0, k=5)
    assert confidence == 0.0


def test_ndcg_perfect(evaluator):
    """Test NDCG with perfect ranking"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 1, 2, 3, 4]

    ndcg = evaluator.calculate_ndcg(retrieved, relevant, k=5)
    assert ndcg == 1.0


def test_ndcg_partial(evaluator):
    """Test NDCG with partial matches"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 2, 4]

    ndcg = evaluator.calculate_ndcg(retrieved, relevant, k=5)
    # Should be between 0 and 1
    assert 0.0 < ndcg <= 1.0


def test_ndcg_no_relevant(evaluator):
    """Test NDCG with no relevant items"""
    retrieved = [0, 1, 2]
    relevant = [5, 6, 7]

    ndcg = evaluator.calculate_ndcg(retrieved, relevant, k=3)
    assert ndcg == 0.0


def test_evaluate_retrieval(evaluator):
    """Test comprehensive evaluation"""
    retrieved = [0, 1, 2, 3, 4]
    relevant = [0, 2, 4, 6]

    results = evaluator.evaluate_retrieval(retrieved, relevant, k=5)

    assert 'precision_at_k' in results
    assert 'recall_at_k' in results
    assert 'f1_at_k' in results
    assert 'mrr' in results
    assert 'ndcg_at_k' in results
    assert 'confidence' in results
    assert results['num_retrieved'] == len(retrieved)
    assert results['num_relevant'] == len(relevant)


def test_compare_retrievals(evaluator):
    """Test comparison of two retrieval methods"""
    retrieval_a = [0, 1, 2, 3, 4]
    retrieval_b = [0, 2, 4, 6, 8]
    relevant = [0, 2, 4, 6]

    comparison = evaluator.compare_retrievals(retrieval_a, retrieval_b, relevant, k=5)

    assert 'method_a' in comparison
    assert 'method_b' in comparison
    assert 'improvements' in comparison
    assert 'precision' in comparison['improvements']
    assert 'recall' in comparison['improvements']
