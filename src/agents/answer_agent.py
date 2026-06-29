"""Answer generation node for LangGraph."""

import logging
from groq import Groq

from src.agents.state import RAGState
from src.config import ANSWER_MODEL, ANSWER_MODEL_TEMP, ANSWER_MODEL_MAX_TOKENS, GROQ_API_KEY

logger = logging.getLogger(__name__)


class AnswerAgent:
    """Generate answers using Groq LLM."""
    
    def __init__(self):
        """Initialize answer agent."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = ANSWER_MODEL
        self.temperature = ANSWER_MODEL_TEMP
        self.max_tokens = ANSWER_MODEL_MAX_TOKENS
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: Original user query
            context: Assembled context from retrieval
            
        Returns:
            Generated answer
        """
        logger.info("Generating answer using Groq...")
        
        system_prompt = """You are a financial analyst assistant specializing in SEC 10-Q filings.

Your task is to answer questions using the provided financial data from SEC filings.

Guidelines:
1. Answer questions directly and accurately based on the provided context
2. Cite sources using [SOURCE: filename] notation
3. If the answer spans multiple sources, cite all relevant sources
4. If the context doesn't contain enough information to answer fully, say so clearly
5. Provide specific figures and data points when available
6. Explain financial metrics and ratios when relevant
7. Be professional and precise in your tone"""
        
        user_message = f"""Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the above context."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
            )
            
            answer = response.content[0].text
            logger.info("✓ Answer generated")
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"


def answer_agent_node(state: RAGState) -> RAGState:
    """
    LangGraph node: generate answer.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with generated answer
    """
    agent = AnswerAgent()
    
    answer = agent.generate(
        query=state["query"],
        context=state.get("context", ""),
    )
    
    state["answer"] = answer
    
    return state
