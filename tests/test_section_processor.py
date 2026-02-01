"""
Tests for SectionProcessor
"""
import pytest
from app.services.section_processor import SectionProcessor, Section


@pytest.fixture
def processor():
    return SectionProcessor()


def test_detect_page_header(processor):
    """Test PAGE X header detection"""
    line = "PAGE 5"
    header = processor._detect_header(line)
    assert header is not None
    assert header['type'] == 'page'
    assert 'Page 5' in header['title']


def test_detect_numbered_section(processor):
    """Test numbered section detection (1., 1.1, etc.)"""
    line = "1. Introduction"
    header = processor._detect_header(line)
    assert header is not None
    assert header['type'] == 'numbered'
    assert 'Introduction' in header['title']

    line = "2.3 Subsection Title"
    header = processor._detect_header(line)
    assert header is not None
    assert header['type'] == 'numbered'
    assert '2.3' in header['title']


def test_detect_capitalized_heading(processor):
    """Test CAPITALIZED HEADING detection"""
    line = "EMPLOYEE BENEFITS"
    header = processor._detect_header(line)
    assert header is not None
    assert header['type'] == 'heading'


def test_no_header_detection(processor):
    """Test regular text is not detected as header"""
    line = "This is just regular text"
    header = processor._detect_header(line)
    assert header is None


def test_is_content_rich(processor):
    """Test content richness validation"""
    # Rich content
    rich = "This is a rich content section with more than sixty characters and at least ten words."
    assert processor.is_content_rich(rich)

    # Too short
    short = "Short"
    assert not processor.is_content_rich(short)

    # Enough chars but too few words
    few_words = "a" * 100
    assert not processor.is_content_rich(few_words)


def test_extract_sections_from_text(processor):
    """Test section extraction from text"""
    text = """PAGE 1

INTRODUCTION

This is the introduction section with sufficient content to pass richness validation.

1. First Section

This is the first section with enough content to be considered rich and meaningful.

2. Second Section

This is the second section, also with sufficient content for validation.
"""

    sections = processor.extract_sections_from_text_direct(
        text,
        document_id="test_doc",
        file_name="test.pdf"
    )

    assert len(sections) > 0
    # Verify section structure
    for section in sections:
        assert section.document_id == "test_doc"
        assert section.file_name == "test.pdf"
        assert len(section.content) >= processor.MIN_CHARS


def test_merge_short_sections(processor):
    """Test merging of short sections"""
    sections = [
        Section(
            title="Section 1",
            content="Short",
            page_number=1,
            section_index=0,
            document_id="doc1",
            file_name="test.pdf",
            section_type="numbered"
        ),
        Section(
            title="Section 2",
            content="Also short",
            page_number=1,
            section_index=1,
            document_id="doc1",
            file_name="test.pdf",
            section_type="numbered"
        ),
        Section(
            title="Section 3",
            content="This is a much longer section with plenty of content" * 5,
            page_number=1,
            section_index=2,
            document_id="doc1",
            file_name="test.pdf",
            section_type="numbered"
        )
    ]

    merged = processor.merge_short_sections(sections, min_length=30)

    # Should merge first two sections
    assert len(merged) < len(sections)
    assert "Section 1" in merged[0].title
    assert "Section 2" in merged[0].title


def test_empty_text(processor):
    """Test handling of empty text"""
    sections = processor.extract_sections_from_text_direct(
        "",
        document_id="test",
        file_name="test.pdf"
    )
    assert len(sections) == 0


def test_no_headers_single_section(processor):
    """Test text with no headers creates single section"""
    text = "This is a document with no headers but sufficient content to be rich and meaningful text."

    sections = processor.extract_sections_from_text_direct(
        text,
        document_id="test",
        file_name="test.pdf"
    )

    # Should create one section for the entire text
    assert len(sections) == 1
    assert "Page 1" in sections[0].title
