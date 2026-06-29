"""RAG retrieval node for LangGraph."""

import logging
from typing import Optional, Dict, Any

from src.agents.state import RAGState
from src.indexing.chroma_store import ChromaDBStore
from src.indexing.embedder import Embedder
from src.indexing.bm25_index import BM25Index
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_ranker import HybridRanker
from src.config import (
    CHROMA_DB_DIR, BM25_INDEX_PATH, BM25_REGISTRY_PATH,
    EMBEDDING_MODEL, VECTOR_RETRIEVAL_TOP_K, BM25_RETRIEVAL_TOP_K,
    HYBRID_FUSION_TOP_K,
)

logger = logging.getLogger(__name__)


class RAGAgent:
    """Perform hybrid retrieval (vector + BM25)."""
    
    _instance = None  # Singleton for efficiency
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(RAGAgent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize RAG agent (lazy initialization)."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG agent...")
        
        # Initialize ChromaDB
        self.chroma_store = ChromaDBStore(db_path=CHROMA_DB_DIR)
        
        # Initialize embedder
        self.embedder = Embedder(model_name=EMBEDDING_MODEL)
        
        # Initialize BM25
        self.bm25_index = BM25Index()
        try:
            self.bm25_index.load(BM25_INDEX_PATH, BM25_REGISTRY_PATH)
        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}. BM25 retrieval will not be available.")
            self.bm25_index = None
        
        # Initialize retrievers
        self.vector_retriever = VectorRetriever(self.chroma_store, self.embedder)
        self.bm25_retriever = BM25Retriever(self.bm25_index) if self.bm25_index else None
        
        # Initialize ranker
        self.hybrid_ranker = HybridRanker()
        
        self._initialized = True
        logger.info("✓ RAG agent initialized")
    
    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Query text
            metadata_filters: Optional metadata filters
            
        Returns:
            List of RankedChunk dicts sorted by relevance
        """
        logger.info(f"RAG retrieval for: {query[:50]}...")
        
        # Vector retrieval
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=VECTOR_RETRIEVAL_TOP_K,
            where_filter=metadata_filters,
        )
        
        # BM25 retrieval
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=BM25_RETRIEVAL_TOP_K,
                where_filter=metadata_filters,
            )
        else:
            logger.warning("BM25 retriever not available, using only vector results")
            bm25_results = []
        
        # Hybrid fusion
        ranked_chunks = self.hybrid_ranker.fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=HYBRID_FUSION_TOP_K,
        )
        
        # Convert to dicts
        results = [chunk.to_dict() for chunk in ranked_chunks]
        
        logger.info(f"✓ Retrieved {len(results)} chunks")
        return results


def rag_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: retrieve documents.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with RAG results
    """
    rag_agent = RAGAgent()
    
    results = rag_agent.retrieve(
        query=state["query"],
        metadata_filters=state.get("metadata_filters"),
    )
    
    state["rag_results"] = results
    
    return state
