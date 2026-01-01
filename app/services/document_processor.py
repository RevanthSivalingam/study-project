import os
import hashlib
from typing import List, Dict, Tuple
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import settings


class DocumentProcessor:
    """Handles PDF document processing, extraction, and chunking"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text from PDF with page-level metadata

        Returns:
            Tuple of (full_text, page_metadata_list)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        full_text = ""
        page_metadata = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
                    page_metadata.append({
                        "page_number": page_num,
                        "text": text,
                        "char_count": len(text)
                    })

        return full_text, page_metadata

    def chunk_document(
        self,
        file_path: str,
        document_type: str = "policy"
    ) -> List[Document]:
        """
        Process PDF and create chunked documents with metadata

        Returns:
            List of LangChain Document objects with metadata
        """
        # Extract text
        full_text, page_metadata = self.extract_text_from_pdf(file_path)

        # Generate document ID
        doc_id = self._generate_document_id(file_path)
        file_name = os.path.basename(file_path)

        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)

        # Create Document objects with metadata
        documents = []
        for idx, chunk in enumerate(chunks):
            # Find which page this chunk primarily comes from
            page_num = self._find_page_for_chunk(chunk, page_metadata)

            doc = Document(
                page_content=chunk,
                metadata={
                    "document_id": doc_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "document_type": document_type,
                    "chunk_id": f"{doc_id}_chunk_{idx}",
                    "chunk_index": idx,
                    "page_number": page_num,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        return documents

    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID based on file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

    def _find_page_for_chunk(self, chunk: str, page_metadata: List[Dict]) -> int:
        """Find the page number that best matches the chunk content"""
        # Simple heuristic: find first page that contains significant portion of chunk
        chunk_start = chunk[:100].strip()  # First 100 chars

        for page_info in page_metadata:
            if chunk_start in page_info["text"]:
                return page_info["page_number"]

        # Default to page 1 if not found
        return 1

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract basic metadata from PDF"""
        with pdfplumber.open(file_path) as pdf:
            return {
                "total_pages": len(pdf.pages),
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "pdf_metadata": pdf.metadata
            }
