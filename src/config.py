"""Configuration management for the hybrid RAG engine."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(DATA_DIR / "docs")))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db")))
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", str(DATA_DIR / "bm25_index.pkl")))
BM25_REGISTRY_PATH = Path(os.getenv("BM25_REGISTRY_PATH", str(DATA_DIR / "bm25_registry.json")))
QNA_DATA_PATH = DATA_DIR / "qna_data.csv"
QNA_DATA_MINI_PATH = DATA_DIR / "qna_data_mini.csv"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LLM Models & API
# ============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QUERY_ANALYZER_MODEL = os.getenv("QUERY_ANALYZER_MODEL", "llama-3.1-8b-instant")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "llama-3.3-70b-versatile")
ANSWER_MODEL_TEMP = float(os.getenv("ANSWER_MODEL_TEMP", 0.7))
ANSWER_MODEL_MAX_TOKENS = int(os.getenv("ANSWER_MODEL_MAX_TOKENS", 1024))

# ============================================================================
# Retrieval Configuration
# ============================================================================
VECTOR_RETRIEVAL_TOP_K = int(os.getenv("VECTOR_RETRIEVAL_TOP_K", 20))
BM25_RETRIEVAL_TOP_K = int(os.getenv("BM25_RETRIEVAL_TOP_K", 20))
HYBRID_FUSION_TOP_K = int(os.getenv("HYBRID_FUSION_TOP_K", 10))
BM25_RRF_K = int(os.getenv("BM25_RRF_K", 60))

# ============================================================================
# Chunking Configuration
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ============================================================================
# Evaluation
# ============================================================================
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", 5))
USE_RAGAS = os.getenv("USE_RAGAS", "true").lower() == "true"

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# Data URLs for Download
# ============================================================================
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/docugami/KG-RAG-datasets/main/sec-10-q/data/v1"

# PDF files to download
PDF_FILES = [
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

QNA_CSV_FILE = "qna_data.csv"
QNA_CSV_MINI_FILE = "qna_data_mini.csv"

# ============================================================================
# SEC 10-Q Section Headings (for section-aware chunking)
# ============================================================================
SEC_SECTION_PATTERNS = [
    r"^ITEM\s+1[\s\.]",
    r"^ITEM\s+2[\s\.]",
    r"^ITEM\s+3[\s\.]",
    r"^ITEM\s+4[\s\.]",
    r"^ITEM\s+5[\s\.]",
    r"^ITEM\s+6[\s\.]",
    r"^Notes\s+to\s+Condensed",
    r"^MANAGEMENT'S DISCUSSION",
    r"^BUSINESS",
    r"^RISK FACTORS",
    r"^FINANCIAL STATEMENTS",
]
