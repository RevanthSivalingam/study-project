"""
Tests for MMRRetriever
"""
import pytest
from unittest.mock import Mock
from app.services.mmr_retriever import MMRRetriever, tokenize_sentences


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings"""
    embeddings = Mock()

    # Mock embed_query to return a simple vector
    embeddings.embed_query.return_value = [0.5, 0.5, 0.5]

    # Mock embed_documents to return vectors
    def embed_docs(texts):
        # Return different vectors for each text
        return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(texts))]

    embeddings.embed_documents.side_effect = embed_docs

    return embeddings


@pytest.fixture
def retriever(mock_embeddings):
    return MMRRetriever(mock_embeddings)


@pytest.fixture
def sample_sentences():
    return [
        "The company offers maternity leave.",
        "Maternity leave duration is 16 weeks.",
        "Employees can take sick leave.",
        "Sick leave is limited to 10 days.",
        "Vacation time must be approved."
    ]


def test_select_sentences_basic(retriever, sample_sentences):
    """Test basic MMR sentence selection"""
    query = "maternity leave"
    selected = retriever.select_sentences(query, sample_sentences, k=3, lambda_param=0.7)

    assert isinstance(selected, list)
    assert len(selected) <= 3
    assert all(s in sample_sentences for s in selected)


def test_select_sentences_k_larger_than_available(retriever, sample_sentences):
    """Test MMR when k > number of sentences"""
    query = "leave policy"
    selected = retriever.select_sentences(query, sample_sentences, k=10, lambda_param=0.7)

    # Should return at most len(sample_sentences)
    assert len(selected) <= len(sample_sentences)


def test_select_sentences_empty_list(retriever):
    """Test MMR with empty sentence list"""
    query = "test query"
    selected = retriever.select_sentences(query, [], k=3, lambda_param=0.7)

    assert selected == []


def test_select_sentences_with_indices(retriever, sample_sentences):
    """Test MMR with index return"""
    query = "maternity leave"
    selected, indices = retriever.select_sentences_with_indices(
        query, sample_sentences, k=3, lambda_param=0.7
    )

    assert len(selected) == len(indices)
    assert all(0 <= idx < len(sample_sentences) for idx in indices)
    # Indices should be sorted
    assert indices == sorted(indices)


def test_select_sentences_lambda_extremes(retriever, sample_sentences):
    """Test MMR with extreme lambda values"""
    query = "leave"

    # Lambda = 1.0 (pure relevance, no diversity)
    selected_pure_relevance = retriever.select_sentences(
        query, sample_sentences, k=3, lambda_param=1.0
    )
    assert len(selected_pure_relevance) <= 3

    # Lambda = 0.0 (pure diversity, no relevance)
    selected_pure_diversity = retriever.select_sentences(
        query, sample_sentences, k=3, lambda_param=0.0
    )
    assert len(selected_pure_diversity) <= 3


def test_rerank_by_relevance(retriever, sample_sentences):
    """Test sentence reranking"""
    query = "maternity leave"
    ranked = retriever.rerank_by_relevance(query, sample_sentences)

    assert isinstance(ranked, list)
    assert len(ranked) == len(sample_sentences)
    # Each item is (sentence, score)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)
    # Should be sorted by score (descending)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_empty_list(retriever):
    """Test reranking with empty list"""
    query = "test"
    ranked = retriever.rerank_by_relevance(query, [])

    assert ranked == []


def test_tokenize_sentences():
    """Test sentence tokenization"""
    text = "This is sentence one. This is sentence two! Is this sentence three?"
    sentences = tokenize_sentences(text)

    assert isinstance(sentences, list)
    assert len(sentences) == 3
    assert "sentence one" in sentences[0]


def test_tokenize_sentences_single(tokenize_sentences):
    """Test tokenization of single sentence"""
    text = "This is a single sentence."
    from app.services.mmr_retriever import tokenize_sentences as tokenize
    sentences = tokenize(text)

    assert len(sentences) == 1


def test_cosine_similarity(retriever):
    """Test cosine similarity calculation"""
    import numpy as np

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])

    # Identical vectors should have similarity 1.0
    sim = retriever._cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 0.001

    # Orthogonal vectors should have similarity 0.0
    vec3 = np.array([0.0, 1.0, 0.0])
    sim = retriever._cosine_similarity(vec1, vec3)
    assert abs(sim) < 0.001


def test_cosine_similarity_zero_vectors(retriever):
    """Test cosine similarity with zero vectors"""
    import numpy as np

    vec1 = np.array([0.0, 0.0, 0.0])
    vec2 = np.array([1.0, 1.0, 1.0])

    sim = retriever._cosine_similarity(vec1, vec2)
    assert sim == 0.0
