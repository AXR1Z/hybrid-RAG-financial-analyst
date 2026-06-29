"""LangGraph orchestrator for hybrid RAG pipeline."""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.state import RAGState
from src.agents.query_analyzer import query_analyzer_node
from src.agents.rag_agent import rag_agent_node
from src.agents.web_search_agent import web_search_node
from src.agents.context_assembler import context_assembler_node
from src.agents.answer_agent import answer_agent_node

logger = logging.getLogger(__name__)


def should_search_web(state: RAGState) -> Literal["web_search", "context_assembly"]:
    """
    Conditional edge: determine if web search is needed.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node: "web_search" or "context_assembly"
    """
    if state.get("needs_web_search", False):
        logger.info("Web search needed, routing to web_search node")
        return "web_search"
    else:
        logger.info("Web search not needed, routing to context_assembly")
        return "context_assembly"


class RAGOrchestrator:
    """Orchestrate the hybrid RAG pipeline using LangGraph."""
    
    def __init__(self):
        """Initialize orchestrator."""
        logger.info("Initializing RAG orchestrator")
        
        # Create graph
        self.workflow = StateGraph(RAGState)
        
        # Add nodes
        self.workflow.add_node("query_analyzer", query_analyzer_node)
        self.workflow.add_node("rag_agent", rag_agent_node)
        self.workflow.add_node("web_search", web_search_node)
        self.workflow.add_node("context_assembler", context_assembler_node)
        self.workflow.add_node("answer_agent", answer_agent_node)
        
        # Add edges
        self.workflow.add_edge("query_analyzer", "rag_agent")
        
        # Conditional edge: web search decision
        self.workflow.add_conditional_edges(
            "rag_agent",
            should_search_web,
            {
                "web_search": "web_search",
                "context_assembly": "context_assembly",
            },
        )
        
        # Web search -> context assembly
        self.workflow.add_edge("web_search", "context_assembly")
        
        # Context assembly -> answer
        self.workflow.add_edge("context_assembly", "answer_agent")
        
        # Answer -> END
        self.workflow.add_edge("answer_agent", END)
        
        # Set entry point
        self.workflow.set_entry_point("query_analyzer")
        
        # Compile graph
        self.app = self.workflow.compile()
        
        logger.info("✓ Orchestrator initialized")
    
    def invoke(self, query: str, metadata_filters=None) -> RAGState:
        """
        Run the RAG pipeline.
        
        Args:
            query: User query
            metadata_filters: Optional metadata filters
            
        Returns:
            Final RAG state with answer
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing query: {query}")
        logger.info(f"{'='*80}\n")
        
        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "metadata_filters": metadata_filters,
            "rag_results": None,
            "web_results": None,
            "context": "",
            "answer": "",
            "sources": [],
            "needs_web_search": False,
            "rag_hint": None,
            "query_analysis": None,
        }
        
        # Run graph
        try:
            final_state = self.app.invoke(initial_state)
            logger.info("\n✓ Pipeline completed successfully\n")
            return final_state
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise


# Singleton instance
_orchestrator = None


def get_orchestrator() -> RAGOrchestrator:
    """Get or create the orchestrator instance (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
