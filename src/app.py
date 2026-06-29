"""Main application module for the hybrid RAG system."""

import logging
from typing import Optional, Dict, Any

from src.agents.orchestrator import get_orchestrator
from src.agents.state import RAGState

logger = logging.getLogger(__name__)


class FinancialAnalystRAG:
    """Main interface for the hybrid RAG financial analyst."""
    
    def __init__(self):
        """Initialize the RAG system."""
        logger.info("Initializing Financial Analyst RAG system...")
        self.orchestrator = get_orchestrator()
        logger.info("✓ System ready")
    
    def query(
        self,
        question: str,
        company: Optional[str] = None,
        quarter: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question about SEC 10-Q filings
            company: Optional company ticker filter
            quarter: Optional quarter filter (Q1-Q4)
            year: Optional year filter
            
        Returns:
            Dict with 'answer', 'sources', and other metadata
        """
        # Build metadata filters
        metadata_filters = {}
        if company:
            metadata_filters["company"] = company
        if quarter:
            metadata_filters["quarter"] = quarter
        if year:
            metadata_filters["year"] = year
        
        # Run orchestrator
        state = self.orchestrator.invoke(
            query=question,
            metadata_filters=metadata_filters if metadata_filters else None,
        )
        
        # Return simplified response
        return {
            "question": question,
            "answer": state["answer"],
            "sources": state["sources"],
            "num_chunks_retrieved": len(state.get("rag_results", [])),
            "num_web_results": len(state.get("web_results", [])),
        }
    
    def batch_query(
        self,
        questions: list,
        **filters,
    ) -> list:
        """
        Query with multiple questions.
        
        Args:
            questions: List of questions
            **filters: Metadata filters (company, quarter, year)
            
        Returns:
            List of results
        """
        results = []
        for i, question in enumerate(questions):
            logger.info(f"[{i+1}/{len(questions)}] Processing question...")
            try:
                result = self.query(question, **filters)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                })
        
        return results


# Create singleton instance
_rag_instance = None


def get_rag_system() -> FinancialAnalystRAG:
    """Get or create the RAG system instance (singleton)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FinancialAnalystRAG()
    return _rag_instance
