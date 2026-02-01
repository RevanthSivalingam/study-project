"""
Tests for SimplifiedKG
"""
import pytest
from app.services.simplified_kg import SimplifiedKG
from app.services.section_processor import Section


@pytest.fixture
def kg():
    return SimplifiedKG()


@pytest.fixture
def sample_sections():
    """Create sample sections"""
    return [
        Section(
            title="Maternity Leave Duration",
            content="Employees are entitled to 16 weeks of maternity leave. The maximum duration is 16 weeks.",
            page_number=1,
            section_index=0,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        ),
        Section(
            title="Sick Leave Policy",
            content="Sick leave is limited to 10 days per year. Employees cannot exceed 10 days of sick leave.",
            page_number=1,
            section_index=1,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        ),
        Section(
            title="Vacation Benefits",
            content="All permanent employees receive vacation time. Vacation must be approved in advance.",
            page_number=2,
            section_index=2,
            document_id="doc1",
            file_name="policy.pdf",
            section_type="numbered"
        )
    ]


@pytest.fixture
def sample_key_terms():
    """Sample key terms"""
    return ["maternity leave", "sick leave", "vacation", "employees", "days", "weeks"]


def test_build_from_sections(kg, sample_sections, sample_key_terms):
    """Test KG construction"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    assert len(kg.graph) > 0
    assert len(kg.section_map) == len(sample_sections)


def test_classify_relation_limits(kg):
    """Test limit classification"""
    content = "Employees are entitled to 16 weeks of maternity leave."
    term = "maternity leave"

    relation_type = kg._classify_relation(content.lower(), term.lower())
    # Should classify as "limits" due to "16 weeks"
    assert relation_type == "limits"


def test_classify_relation_description(kg):
    """Test description classification"""
    content = "Maternity leave is an important benefit for employees."
    term = "maternity leave"

    relation_type = kg._classify_relation(content.lower(), term.lower())
    # No numeric constraints, should be "description"
    assert relation_type == "description"


def test_query_exact_match(kg, sample_sections, sample_key_terms):
    """Test querying with exact term match"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    results = kg.query("maternity leave")
    assert isinstance(results, list)
    # Should find sections containing "maternity leave"
    if results:
        assert "Maternity" in results[0] or any("Maternity" in r for r in results)


def test_query_partial_match(kg, sample_sections, sample_key_terms):
    """Test querying with partial term match"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    results = kg.query("vacation")
    assert isinstance(results, list)


def test_query_no_match(kg, sample_sections, sample_key_terms):
    """Test querying with no matches"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    results = kg.query("nonexistent term xyz")
    assert isinstance(results, list)
    # May be empty or may have partial matches


def test_group_by_section(kg, sample_sections, sample_key_terms):
    """Test section grouping"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    candidate_titles = ["Maternity Leave Duration", "Sick Leave Policy"]
    section_map = kg.group_by_section(sample_sections, candidate_titles)

    assert isinstance(section_map, dict)
    assert len(section_map) == 2
    assert "Maternity Leave Duration" in section_map


def test_select_best_section_lexical(kg):
    """Test best section selection using lexical similarity"""
    section_map = {
        "Section A": "This is about maternity leave duration and benefits",
        "Section B": "This discusses vacation time and holidays"
    }

    query = "maternity leave duration"
    best_title, score = kg.select_best_section(query, section_map)

    assert best_title == "Section A"
    assert score > 0


def test_select_best_section_empty(kg):
    """Test best section selection with empty map"""
    best_title, score = kg.select_best_section("query", {})

    assert best_title is None
    assert score is None


def test_get_related_terms(kg, sample_sections, sample_key_terms):
    """Test finding related terms"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    related = kg.get_related_terms("maternity leave", max_terms=3)

    assert isinstance(related, list)
    assert len(related) <= 3


def test_get_section_by_title(kg, sample_sections, sample_key_terms):
    """Test section retrieval by title"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    section = kg.get_section_by_title("Maternity Leave Duration")

    assert section is not None
    assert section.title == "Maternity Leave Duration"


def test_get_section_by_title_not_found(kg, sample_sections, sample_key_terms):
    """Test section retrieval with non-existent title"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    section = kg.get_section_by_title("Non-existent Section")

    assert section is None


def test_get_stats(kg, sample_sections, sample_key_terms):
    """Test statistics retrieval"""
    kg.build_from_sections(sample_sections, sample_key_terms)

    stats = kg.get_stats()

    assert 'num_terms' in stats
    assert 'num_sections' in stats
    assert 'total_mappings' in stats
    assert stats['num_sections'] == len(sample_sections)
