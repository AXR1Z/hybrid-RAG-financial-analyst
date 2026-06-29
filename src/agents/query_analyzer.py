"""Query analysis node for LangGraph."""

import json
import logging
from typing import Dict, Any, Optional, List
from groq import Groq

from src.config import QUERY_ANALYZER_MODEL, GROQ_API_KEY
from src.agents.state import RAGState

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze user query to extract metadata and determine retrieval strategy."""
    
    def __init__(self):
        """Initialize query analyzer."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = QUERY_ANALYZER_MODEL
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract companies, quarters, and determine web search need.
        
        Args:
            query: User query
            
        Returns:
            Dict with keys: companies, quarters, year_range, needs_web_search, rag_hint
        """
        logger.info(f"Analyzing query: {query[:50]}...")
        
        system_prompt = """You are a financial analyst query analyzer. Analyze the user's question and extract:
1. companies: List of company tickers (AAPL, AMZN, INTC, MSFT, NVDA) mentioned
2. quarters: List of quarters mentioned (e.g., Q1, Q2, Q3, Q4) or empty if not specified
3. years: List of years mentioned (e.g., 2022, 2023) or empty if not specified
4. needs_web_search: Boolean - true if query asks about current events, news, or external data not in SEC filings
5. rag_hint: One of "single-doc-single-chunk", "single-doc-multi-chunk", "multi-doc", or null if unclear

Return a JSON object with these exact keys."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Query: {query}",
                    }
                ],
            )
            
            response_text = response.content[0].text
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (in case there's extra text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    analysis = {}
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {response_text}")
                analysis = {}
            
            logger.info(f"Query analysis: {analysis}")
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {}
    
    def build_metadata_filter(self, analysis: Dict[str, Any]) -> Optional[Dict]:
        """
        Build ChromaDB/BM25 metadata filter from analysis.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Dict with metadata filters, or None if no filters
        """
        filters = {}
        
        # Add company filter
        companies = analysis.get("companies", [])
        if companies:
            # Map company names to metadata values if needed
            filters["company"] = companies
        
        # Add quarter filter
        quarters = analysis.get("quarters", [])
        if quarters:
            filters["quarter"] = quarters
        
        # Add year filter
        years = analysis.get("years", [])
        if years:
            filters["year"] = years
        
        return filters if filters else None


def query_analyzer_node(state: RAGState) -> RAGState:
    """
    LangGraph node: analyze query.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with query analysis results
    """
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(state["query"])
    
    # Update state
    state["query_analysis"] = analysis
    state["needs_web_search"] = analysis.get("needs_web_search", False)
    state["rag_hint"] = analysis.get("rag_hint")
    state["metadata_filters"] = analyzer.build_metadata_filter(analysis)
    
    return state
