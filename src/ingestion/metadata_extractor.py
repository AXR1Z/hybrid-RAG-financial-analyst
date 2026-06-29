"""Extract metadata from PDF filenames."""

import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def extract_metadata_from_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Extract metadata from SEC 10-Q PDF filename.
    
    Expected format: "{year} Q{quarter} {ticker}.pdf"
    Example: "2023 Q3 AAPL.pdf"
    
    Args:
        filename: The PDF filename
        
    Returns:
        Dict with keys: year, quarter, company, ticker, doc_name, or None if no match
    """
    # Pattern: YYYY Qn TICKER.pdf
    pattern = r"^(\d{4})\s+Q(\d)\s+([A-Z]{3,4})\.pdf$"
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    year_str, quarter_str, ticker = match.groups()
    year = int(year_str)
    quarter = int(quarter_str)
    
    # Validate quarter
    if quarter < 1 or quarter > 4:
        return None
    
    # Map ticker to company name
    ticker_to_company = {
        "AAPL": "Apple",
        "AMZN": "Amazon",
        "INTC": "Intel",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
    }
    
    company = ticker_to_company.get(ticker)
    if not company:
        return None
    
    # Calculate approximate filing date (Q3 = ~Aug/Sep, etc.)
    quarter_months = {1: 2, 2: 5, 3: 8, 4: 11}
    filing_month = quarter_months[quarter]
    
    return {
        "year": year,
        "quarter": quarter,
        "company": company,
        "ticker": ticker,
        "doc_name": filename.replace(".pdf", ""),
        "filing_period": f"{year}Q{quarter}",
        "filing_month": filing_month,
    }


def extract_metadata_from_path(pdf_path: Path) -> Optional[Dict[str, any]]:
    """
    Extract metadata from a PDF file path.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dict with metadata or None
    """
    if not pdf_path.exists():
        return None
    
    filename = pdf_path.name
    metadata = extract_metadata_from_filename(filename)
    
    if metadata:
        metadata["file_path"] = str(pdf_path)
        metadata["file_size_bytes"] = pdf_path.stat().st_size
    
    return metadata
