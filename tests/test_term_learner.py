"""
Tests for TermLearner
"""
import pytest
import tempfile
import shutil
from app.services.term_learner import TermLearner
from app.services.section_processor import Section


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def term_learner(temp_dir):
    return TermLearner(persist_directory=temp_dir)


@pytest.fixture
def sample_sections():
    """Create sample sections for testing"""
    return [
        Section(
            title="Maternity Leave",
            content="Employees are entitled to maternity leave of 16 weeks. The leave can be taken before or after delivery.",
            page_number=1,
            section_index=0,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        ),
        Section(
            title="Sick Leave",
            content="Sick leave is provided for medical reasons. Employees can take up to 10 days of sick leave per year.",
            page_number=1,
            section_index=1,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        ),
        Section(
            title="Vacation Policy",
            content="All employees receive vacation time. Vacation days must be approved by the manager in advance.",
            page_number=2,
            section_index=2,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        )
    ]


def test_learn_generic_terms(term_learner, sample_sections):
    """Test generic term learning"""
    generic_terms = term_learner.learn_generic_terms(sample_sections)

    assert isinstance(generic_terms, set)
    assert len(generic_terms) > 0
    assert term_learner.is_fitted


def test_learn_key_terms(term_learner, sample_sections):
    """Test key term extraction"""
    # First learn generic terms
    term_learner.learn_generic_terms(sample_sections)

    # Then learn key terms
    key_terms = term_learner.learn_key_terms(sample_sections, top_n=10)

    assert isinstance(key_terms, list)
    assert len(key_terms) > 0
    assert len(key_terms) <= 10


def test_normalize_query(term_learner, sample_sections):
    """Test query normalization"""
    # Learn terms first
    term_learner.learn_generic_terms(sample_sections)

    query = "what is the policy for maternity leave"
    normalized = term_learner.normalize_query(query)

    # Should remove some generic terms
    assert isinstance(normalized, str)
    # Normalized should be different from original (some terms removed)
    # But should contain key terms like 'maternity' and 'leave'
    assert 'maternity' in normalized.lower()


def test_normalize_empty_result_fallback(term_learner, sample_sections):
    """Test that normalization returns original if everything is filtered"""
    term_learner.learn_generic_terms(sample_sections)

    # Query with only generic terms
    query = "the a an is"
    normalized = term_learner.normalize_query(query)

    # Should return original query if normalization removes everything
    assert normalized == query


def test_extract_key_terms_from_text(term_learner, sample_sections):
    """Test key term extraction from specific text"""
    # Learn terms first
    term_learner.learn_generic_terms(sample_sections)

    text = "Maternity leave policy for employees"
    key_terms = term_learner.extract_key_terms_from_text(text, top_n=3)

    assert isinstance(key_terms, list)
    assert len(key_terms) <= 3


def test_save_and_load(term_learner, sample_sections, temp_dir):
    """Test model persistence"""
    # Learn terms
    term_learner.learn_generic_terms(sample_sections)
    term_learner.learn_key_terms(sample_sections)

    # Save
    term_learner.save()

    # Create new learner and load
    new_learner = TermLearner(persist_directory=temp_dir)
    loaded = new_learner.load()

    assert loaded is True
    assert new_learner.is_fitted
    assert len(new_learner.generic_terms) == len(term_learner.generic_terms)
    assert len(new_learner.key_terms) == len(term_learner.key_terms)


def test_load_nonexistent_model(term_learner):
    """Test loading when no model exists"""
    loaded = term_learner.load()
    assert loaded is False


def test_get_stats(term_learner, sample_sections):
    """Test statistics retrieval"""
    # Before fitting
    stats = term_learner.get_stats()
    assert stats['is_fitted'] is False

    # After fitting
    term_learner.learn_generic_terms(sample_sections)
    term_learner.learn_key_terms(sample_sections)

    stats = term_learner.get_stats()
    assert stats['is_fitted'] is True
    assert stats['num_generic_terms'] > 0
    assert stats['num_key_terms'] > 0
    assert 'generic_terms_sample' in stats
    assert 'key_terms_sample' in stats


def test_get_term_frequencies(term_learner, sample_sections):
    """Test term frequency counting"""
    freqs = term_learner.get_term_frequencies(sample_sections)

    assert isinstance(freqs, dict)
    assert len(freqs) > 0
    # Common words should appear
    assert 'leave' in freqs or 'employees' in freqs
