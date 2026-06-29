"""Context assembly node for LangGraph."""

import logging
from typing import List, Dict, Any

from src.agents.state import RAGState

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assemble retrieved context for LLM."""
    
    def __init__(self, max_chunks: int = 10, max_web_results: int = 5):
        """
        Initialize assembler.
        
        Args:
            max_chunks: Maximum number of RAG chunks to include
            max_web_results: Maximum number of web results to include
        """
        self.max_chunks = max_chunks
        self.max_web_results = max_web_results
    
    def assemble(
        self,
        rag_results: List[Dict[str, Any]],
        web_results: List[Dict[str, str]],
    ) -> tuple:
        """
        Assemble context from retrieved results.
        
        Args:
            rag_results: RAG retrieval results (can be None or empty list)
            web_results: Web search results (can be None or empty list)
            
        Returns:
            Tuple of (context_string, sources_list)
        """
        # Handle None values
        rag_results = rag_results or []
        web_results = web_results or []
        
        context_parts = []
        sources = []
        
        # Add RAG results
        if rag_results:
            context_parts.append("## SEC 10-Q Financial Data\n")
            
            for i, chunk in enumerate(rag_results[:self.max_chunks], 1):
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                
                # Extract source info
                company = metadata.get("company", "")
                ticker = metadata.get("ticker", "")
                filing_period = metadata.get("filing_period", "")
                doc_name = metadata.get("doc_name", "")
                section = chunk.get("section_name", "")
                
                source_cite = f"{doc_name}.pdf"
                if section:
                    source_cite += f" ({section})"
                
                context_parts.append(f"[{i}] {text}\n")
                context_parts.append(f"SOURCE: {source_cite}\n\n")
                
                sources.append({
                    "type": "filing",
                    "source": source_cite,
                    "company": company,
                    "ticker": ticker,
                    "filing_period": filing_period,
                })
        
        # Add web results
        if web_results:
            context_parts.append("\n## Recent Web Information\n")
            
            for i, result in enumerate(web_results[:self.max_web_results], 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("url", "")
                
                context_parts.append(f"[WEB-{i}] {title}\n")
                context_parts.append(f"{snippet}\n")
                context_parts.append(f"SOURCE: {url}\n\n")
                
                sources.append({
                    "type": "web",
                    "source": url,
                    "title": title,
                })
        
        context = "\n".join(context_parts)
        
        logger.info(f"Assembled context from {len(rag_results or [])} RAG chunks + {len(web_results or [])} web results")
        
        return context, sources


def context_assembler_node(state: RAGState) -> RAGState:
    """
    LangGraph node: assemble context.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with assembled context
    """
    assembler = ContextAssembler()
    
    context, sources = assembler.assemble(
        rag_results=state.get("rag_results", []),
        web_results=state.get("web_results", []),
    )
    
    state["context"] = context
    state["sources"] = [s["source"] for s in sources]
    
    return state
