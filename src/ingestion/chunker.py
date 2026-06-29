"""Intelligent chunking of SEC 10-Q documents."""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    
    chunk_id: str
    text: str
    section_name: Optional[str] = None
    content_type: str = "text"  # 'text' or 'table'
    metadata: Optional[Dict] = None
    page_num: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "section_name": self.section_name,
            "content_type": self.content_type,
            "metadata": self.metadata or {},
            "page_num": self.page_num,
        }


class DocumentChunker:
    """Chunk SEC 10-Q documents using section-aware + recursive character splitting."""
    
    # SEC section patterns (regex)
    SEC_SECTION_PATTERNS = [
        (r"^ITEM\s+1[\.\s]", "ITEM 1: Business"),
        (r"^ITEM\s+2[\.\s]", "ITEM 2: MD&A"),
        (r"^ITEM\s+3[\.\s]", "ITEM 3: QC"),
        (r"^ITEM\s+4[\.\s]", "ITEM 4: Controls"),
        (r"^ITEM\s+5[\.\s]", "ITEM 5: Market"),
        (r"^ITEM\s+6[\.\s]", "ITEM 6: Exhibits"),
        (r"^Notes\s+to\s+Condensed", "Notes to Financials"),
        (r"^MANAGEMENT'S\s+DISCUSSION", "Management Discussion & Analysis"),
        (r"^FINANCIAL\s+STATEMENTS", "Financial Statements"),
        (r"^RISK\s+FACTORS", "Risk Factors"),
        (r"^CONDENSED\s+CONSOLIDATED", "Consolidated Statements"),
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (characters)
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to include
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Compile patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), name)
            for pattern, name in self.SEC_SECTION_PATTERNS
        ]
    
    def split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into sections based on SEC headings.
        
        Args:
            text: Full document text
            
        Returns:
            List of (section_name, section_text) tuples
        """
        sections = []
        current_section = "Preamble"
        current_text = ""
        
        lines = text.split("\n")
        
        for line in lines:
            # Check if line matches any section heading
            matched_section = None
            for pattern, section_name in self.compiled_patterns:
                if pattern.search(line):
                    matched_section = section_name
                    break
            
            if matched_section:
                # Save previous section
                if current_text.strip():
                    sections.append((current_section, current_text))
                
                # Start new section
                current_section = matched_section
                current_text = line + "\n"
            else:
                current_text += line + "\n"
        
        # Don't forget last section
        if current_text.strip():
            sections.append((current_section, current_text))
        
        return sections
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size
            
        Returns:
            List of text chunks
        """
        size = chunk_size or self.chunk_size
        chunks = []
        
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end]
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start += size - self.chunk_overlap
        
        return chunks
    
    def chunk_document(
        self,
        text: str,
        doc_metadata: Dict,
        content_type: str = "text",
    ) -> List[Chunk]:
        """
        Chunk a full document using section-aware + recursive splitting.
        
        Args:
            text: Full document text
            doc_metadata: Document metadata (company, quarter, etc.)
            content_type: 'text', 'table', or 'mixed'
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_counter = 0
        
        # Phase 1: Split by sections
        sections = self.split_by_sections(text)
        
        logger.info(f"Split document into {len(sections)} sections")
        
        # Phase 2: Chunk each section
        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            
            # If section is small enough, keep as-is
            if len(section_text) <= self.chunk_size * 2:
                chunk_id = f"{doc_metadata['ticker']}_{doc_metadata['filing_period']}_s{chunk_counter}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=section_text.strip(),
                    section_name=section_name,
                    content_type=content_type,
                    metadata=doc_metadata,
                ))
                chunk_counter += 1
            else:
                # Split section further using recursive character splitter
                sub_chunks = self.chunk_text(section_text, self.chunk_size)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_id = f"{doc_metadata['ticker']}_{doc_metadata['filing_period']}_s{chunk_counter}_c{i}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=sub_chunk.strip(),
                        section_name=section_name,
                        content_type=content_type,
                        metadata=doc_metadata,
                    ))
                chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks


def chunk_document(
    text: str,
    doc_metadata: Dict,
    content_type: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """Convenience function to chunk a document."""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(text, doc_metadata, content_type)
