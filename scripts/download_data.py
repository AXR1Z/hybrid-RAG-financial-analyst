"""Download datasets from GitHub."""

import logging
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, save_path: Path, timeout: int = 60) -> bool:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        save_path: Path to save file
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get("content-length", 0))
        
        # Download with progress bar
        with open(save_path, "wb") as f:
            if total_size:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        logger.info(f"✓ Downloaded to {save_path}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to download {url}: {e}")
        return False


def download_sec_10q_data(data_dir: Path, github_base: str) -> bool:
    """
    Download SEC 10-Q PDFs and QnA CSV from GitHub.
    
    Args:
        data_dir: Directory to save data
        github_base: Base GitHub URL
        
    Returns:
        True if all downloads successful
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = data_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # PDF files
    pdf_files = [
        "2022 Q3 AAPL.pdf",
        "2022 Q3 AMZN.pdf",
        "2022 Q3 INTC.pdf",
        "2022 Q3 MSFT.pdf",
        "2022 Q3 NVDA.pdf",
        "2023 Q1 AAPL.pdf",
        "2023 Q1 AMZN.pdf",
        "2023 Q1 INTC.pdf",
        "2023 Q1 MSFT.pdf",
        "2023 Q1 NVDA.pdf",
        "2023 Q2 AAPL.pdf",
        "2023 Q2 AMZN.pdf",
        "2023 Q2 INTC.pdf",
        "2023 Q2 MSFT.pdf",
        "2023 Q2 NVDA.pdf",
        "2023 Q3 AAPL.pdf",
        "2023 Q3 AMZN.pdf",
        "2023 Q3 INTC.pdf",
        "2023 Q3 MSFT.pdf",
        "2023 Q3 NVDA.pdf",
    ]
    
    # CSV files
    csv_files = [
        "qna_data.csv",
        "qna_data_mini.csv",
    ]
    
    success_count = 0
    total_files = len(pdf_files) + len(csv_files)
    
    logger.info(f"Downloading {len(pdf_files)} PDFs...")
    for pdf_file in pdf_files:
        url = f"{github_base}/docs/{pdf_file}"
        save_path = docs_dir / pdf_file
        
        if save_path.exists():
            logger.info(f"⊘ Already exists: {pdf_file}")
            success_count += 1
        elif download_file(url, save_path):
            success_count += 1
    
    logger.info(f"Downloading {len(csv_files)} CSV files...")
    for csv_file in csv_files:
        url = f"{github_base}/{csv_file}"
        save_path = data_dir / csv_file
        
        if save_path.exists():
            logger.info(f"⊘ Already exists: {csv_file}")
            success_count += 1
        elif download_file(url, save_path):
            success_count += 1
    
    logger.info(f"\n✓ Download complete: {success_count}/{total_files} files")
    return success_count == total_files


if __name__ == "__main__":
    from src.config import DATA_DIR, GITHUB_RAW_BASE
    
    success = download_sec_10q_data(DATA_DIR, GITHUB_RAW_BASE)
    exit(0 if success else 1)
