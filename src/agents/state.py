"""LangGraph state definition for RAG pipeline."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from typing_extensions import TypedDict


class RAGState(TypedDict):
    """State passed through LangGraph."""
    
    # Input
    query: str
    
    # Configuration
    metadata_filters: Optional[Dict[str, Any]]
    
    # Retrieval results
    rag_results: Optional[List[Dict[str, Any]]]  # List of retrieved chunks with scores
    
    # Web search results
    web_results: Optional[List[Dict[str, str]]]  # List of {title, snippet, url}
    
    # Context assembly
    context: str  # Formatted context for LLM
    
    # Answer generation
    answer: str  # Generated answer
    sources: List[str]  # List of source citations
    
    # Intermediate flags
    needs_web_search: bool
    rag_hint: Optional[str]  # Hint for RAG type (single-doc, multi-doc, etc)
    
    # Metadata for tracing
    query_analysis: Optional[Dict]  # Results from query analyzer
