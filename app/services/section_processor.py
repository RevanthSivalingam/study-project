"""
Section-based document processing with header detection.

This processor extracts meaningful sections from documents using header patterns
instead of fixed-size chunking, enabling more semantic and context-preserving retrieval.
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pdfplumber


@dataclass
class Section:
    """Represents a document section with metadata"""
    title: str
    content: str
    page_number: int
    section_index: int
    document_id: str
    file_name: str
    section_type: str  # "page", "numbered", "heading"

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "title": self.title,
            "content": self.content,
            "page_number": self.page_number,
            "section_index": self.section_index,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "section_type": self.section_type
        }


class SectionProcessor:
    """
    Processes documents into semantic sections based on header detection.

    Detects headers using multiple patterns:
    - PAGE X
    - 1. Title
    - 1.1 Subtitle
    - CAPITALIZED HEADINGS
    """

    # Header detection patterns
    PAGE_PATTERN = re.compile(r'^(?:PAGE\s+)?(\d+)[\s:\-]*(.*)$', re.IGNORECASE)
    NUMBERED_SECTION = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
    CAPITALIZED_HEADING = re.compile(r'^([A-Z][A-Z\s]{3,})$')  # At least 4 caps chars

    # Content richness thresholds
    MIN_CHARS = 60
    MIN_WORDS = 10

    def __init__(self):
        self.sections = []

    def extract_sections_from_pdf(
        self,
        file_path: str,
        document_id: str
    ) -> List[Section]:
        """
        Extract sections from a PDF using header detection.

        Args:
            file_path: Path to the PDF file
            document_id: Unique identifier for the document

        Returns:
            List of Section objects
        """
        sections = []
        file_name = file_path.split('/')[-1]

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                # Extract sections from this page's text
                page_sections = self._extract_sections_from_text(
                    text,
                    page_num,
                    document_id,
                    file_name
                )
                sections.extend(page_sections)

        # Assign section indices
        for idx, section in enumerate(sections):
            section.section_index = idx

        # Filter out non-rich sections
        rich_sections = [s for s in sections if self.is_content_rich(s.content)]

        return rich_sections

    def _extract_sections_from_text(
        self,
        text: str,
        page_num: int,
        document_id: str,
        file_name: str
    ) -> List[Section]:
        """
        Extract sections from a single page's text.

        Uses header patterns to identify section boundaries.
        """
        sections = []
        lines = text.split('\n')

        current_section_title = None
        current_section_content = []
        current_section_type = None
        section_start_idx = 0

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if this line is a header
            header_info = self._detect_header(line)

            if header_info:
                # Save previous section if exists
                if current_section_title and current_section_content:
                    content = '\n'.join(current_section_content).strip()
                    if content:
                        sections.append(Section(
                            title=current_section_title,
                            content=content,
                            page_number=page_num,
                            section_index=0,  # Will be set later
                            document_id=document_id,
                            file_name=file_name,
                            section_type=current_section_type
                        ))

                # Start new section
                current_section_title = header_info['title']
                current_section_type = header_info['type']
                current_section_content = []
                section_start_idx = idx
            else:
                # Add to current section content
                current_section_content.append(line)

        # Save last section
        if current_section_title and current_section_content:
            content = '\n'.join(current_section_content).strip()
            if content:
                sections.append(Section(
                    title=current_section_title,
                    content=content,
                    page_number=page_num,
                    section_index=0,
                    document_id=document_id,
                    file_name=file_name,
                    section_type=current_section_type
                ))

        # If no sections found, treat entire page as one section
        if not sections and text.strip():
            sections.append(Section(
                title=f"Page {page_num}",
                content=text.strip(),
                page_number=page_num,
                section_index=0,
                document_id=document_id,
                file_name=file_name,
                section_type="page"
            ))

        return sections

    def _detect_header(self, line: str) -> Optional[Dict[str, str]]:
        """
        Detect if a line is a header and return header info.

        Returns:
            Dict with 'title' and 'type' keys, or None if not a header
        """
        # Pattern 1: PAGE X or just a number
        match = self.PAGE_PATTERN.match(line)
        if match and len(line) < 50:  # Headers should be short
            page_num, title = match.groups()
            title = title.strip() if title else f"Page {page_num}"
            return {"title": title, "type": "page"}

        # Pattern 2: Numbered sections (1., 1.1, etc.)
        match = self.NUMBERED_SECTION.match(line)
        if match:
            number, title = match.groups()
            return {"title": f"{number} {title}", "type": "numbered"}

        # Pattern 3: CAPITALIZED HEADINGS
        match = self.CAPITALIZED_HEADING.match(line)
        if match and len(line) < 100:  # Headers should be reasonably short
            return {"title": line, "type": "heading"}

        return None

    def is_content_rich(self, content: str) -> bool:
        """
        Validate that content is rich enough to be useful.

        Criteria:
        - At least MIN_CHARS characters
        - At least MIN_WORDS words

        Args:
            content: The content to validate

        Returns:
            True if content meets richness criteria
        """
        if not content:
            return False

        char_count = len(content.strip())
        word_count = len(content.split())

        return char_count >= self.MIN_CHARS and word_count >= self.MIN_WORDS

    def extract_sections_from_text_direct(
        self,
        text: str,
        document_id: str,
        file_name: str,
        page_num: int = 1
    ) -> List[Section]:
        """
        Extract sections from raw text (not from PDF).
        Useful for testing or when text is already extracted.

        Args:
            text: The text to process
            document_id: Unique identifier for the document
            file_name: Name of the source file
            page_num: Page number (default 1)

        Returns:
            List of Section objects
        """
        sections = self._extract_sections_from_text(
            text,
            page_num,
            document_id,
            file_name
        )

        # Assign section indices
        for idx, section in enumerate(sections):
            section.section_index = idx

        # Filter out non-rich sections
        rich_sections = [s for s in sections if self.is_content_rich(s.content)]

        return rich_sections

    def merge_short_sections(
        self,
        sections: List[Section],
        min_length: int = 200
    ) -> List[Section]:
        """
        Merge consecutive short sections to ensure minimum content length.

        Args:
            sections: List of sections to process
            min_length: Minimum character length for a section

        Returns:
            List of merged sections
        """
        if not sections:
            return []

        merged = []
        current = sections[0]

        for i in range(1, len(sections)):
            next_section = sections[i]

            # If current section is too short, merge with next
            if len(current.content) < min_length:
                current.content = f"{current.content}\n\n{next_section.content}"
                current.title = f"{current.title} / {next_section.title}"
            else:
                merged.append(current)
                current = next_section

        # Add last section
        merged.append(current)

        # Re-index
        for idx, section in enumerate(merged):
            section.section_index = idx

        return merged
