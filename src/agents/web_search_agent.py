"""Web search node for LangGraph."""

import logging
from typing import List, Dict

from src.agents.state import RAGState

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Perform web search using DuckDuckGo."""
    
    def __init__(self):
        """Initialize web search agent."""
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
        except ImportError:
            logger.warning("duckduckgo_search not available")
            self.ddgs = None
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of {title, snippet, url}
        """
        if not self.ddgs:
            logger.warning("Web search not available")
            return []
        
        logger.info(f"Web search for: {query}")
        
        try:
            results = self.ddgs.text(query, max_results=max_results)
            
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "rank": i + 1,
                })
            
            logger.info(f"✓ Found {len(formatted_results)} web results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []


def web_search_node(state: RAGState) -> RAGState:
    """
    LangGraph node: web search (conditional).
    
    Only runs if state["needs_web_search"] is True.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with web results
    """
    if not state.get("needs_web_search", False):
        logger.info("Web search not needed, skipping")
        state["web_results"] = []
        return state
    
    agent = WebSearchAgent()
    results = agent.search(query=state["query"], max_results=5)
    
    state["web_results"] = results
    
    return state
