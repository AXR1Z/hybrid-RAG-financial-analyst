"""PDF parsing and content extraction."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PDFPage:
    """Represents a single page from a PDF."""
    
    def __init__(
        self,
        page_num: int,
        text: str,
        content_type: str = "text",  # 'text', 'table', 'mixed'
        metadata: Optional[Dict] = None,
    ):
        self.page_num = page_num
        self.text = text
        self.content_type = content_type
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"PDFPage(num={self.page_num}, type={self.content_type}, len={len(self.text)})"


class PDFParser:
    """Parse SEC 10-Q PDFs and extract text content."""
    
    # Thresholds for content type detection
    TABLE_THRESHOLD = 0.30  # 30% of chars should be in table cells to flag as table
    
    def __init__(self):
        pass
    
    def detect_content_type(self, page: fitz.Page) -> str:
        """
        Detect if a page is primarily table, text, or mixed.
        
        Uses heuristics: if >30% of extractable chars are in table cells, mark as table.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            'table', 'text', or 'mixed'
        """
        try:
            # Try to extract tables
            tables = page.find_tables()
            if not tables:
                return "text"
            
            # Extract text from table cells
            table_text = ""
            for table in tables.tables:
                for row in table.rows:
                    for cell in row:
                        table_text += cell.get_text()
            
            # Extract all text
            all_text = page.get_text()
            
            if not all_text:
                return "table" if table_text else "text"
            
            table_char_ratio = len(table_text) / len(all_text)
            
            if table_char_ratio > self.TABLE_THRESHOLD:
                return "table"
            elif table_char_ratio > 0.1:
                return "mixed"
            else:
                return "text"
        except Exception as e:
            logger.warning(f"Error detecting content type: {e}. Defaulting to 'text'.")
            return "text"
    
    def parse_pdf(self, pdf_path: Path) -> List[PDFPage]:
        """
        Parse a PDF file and extract pages with text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PDFPage objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        pages = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    
                    # Skip pages with minimal content
                    if len(text.strip()) < 50:
                        logger.debug(f"Skipping page {page_num} (too short)")
                        continue
                    
                    # Detect content type
                    content_type = self.detect_content_type(page)
                    
                    pages.append(PDFPage(
                        page_num=page_num,
                        text=text,
                        content_type=content_type,
                    ))
            
            logger.info(f"Parsed {len(pages)} pages from {pdf_path.name}")
            return pages
        
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract all text from a PDF.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            Concatenated text from all pages
        """
        pages = self.parse_pdf(pdf_path)
        return "\n\n--- PAGE BREAK ---\n\n".join(page.text for page in pages)


def parse_pdf_to_pages(pdf_path: Path) -> List[PDFPage]:
    """Convenience function to parse a PDF."""
    parser = PDFParser()
    return parser.parse_pdf(pdf_path)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Convenience function to extract text from PDF."""
    parser = PDFParser()
    return parser.extract_text(pdf_path)
