"""RAGAS evaluation metrics for RAG system."""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from src.config import DATA_DIR, USE_RAGAS

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system using RAGAS metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.use_ragas = USE_RAGAS
        
        if self.use_ragas:
            try:
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
                self.answer_relevancy = answer_relevancy
                self.context_precision = context_precision
                self.context_recall = context_recall
                self.faithfulness = faithfulness
            except ImportError:
                logger.warning("RAGAS not available, using simplified metrics")
                self.use_ragas = False
    
    def evaluate_single(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single query-answer pair.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            Dict with metric scores
        """
        if self.use_ragas:
            return self._evaluate_ragas(query, answer, context, ground_truth)
        else:
            return self._evaluate_simple(query, answer, context, ground_truth)
    
    def _evaluate_ragas(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate using RAGAS metrics."""
        metrics = {}
        
        try:
            # Answer relevancy: is the answer relevant to the question?
            metrics["answer_relevancy"] = self.answer_relevancy.score(
                {"question": query, "answer": answer}
            )
            
            # Faithfulness: is the answer faithful to the context?
            metrics["faithfulness"] = self.faithfulness.score(
                {"question": query, "answer": answer, "contexts": [context]}
            )
            
            # Context precision: is the retrieved context relevant?
            metrics["context_precision"] = self.context_precision.score(
                {
                    "question": query,
                    "contexts": [context],
                    "ground_truth": ground_truth or query,
                }
            )
            
            # Context recall: was all relevant information retrieved?
            if ground_truth:
                metrics["context_recall"] = self.context_recall.score(
                    {
                        "question": query,
                        "contexts": [context],
                        "ground_truth": ground_truth,
                    }
                )
        except Exception as e:
            logger.warning(f"Error computing RAGAS metrics: {e}")
            metrics = self._evaluate_simple(query, answer, context, ground_truth)
        
        return metrics
    
    def _evaluate_simple(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate using simple heuristics."""
        metrics = {}
        
        # Answer relevancy: do query and answer share keywords?
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        metrics["answer_relevancy"] = min(overlap * 2, 1.0)
        
        # Faithfulness: does answer mention the context?
        context_length = len(context)
        answer_length = len(answer)
        metrics["faithfulness"] = min(answer_length / max(context_length, 1), 1.0)
        
        # Context precision: context length relative to answer
        metrics["context_precision"] = min(context_length / max(answer_length, 1), 1.0)
        
        # Context recall: keyword overlap with ground truth
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            context_words = set(context.lower().split())
            overlap = len(gt_words & context_words) / max(len(gt_words), 1)
            metrics["context_recall"] = overlap
        
        return metrics
    
    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of results.
        
        Args:
            results: List of {query, answer, context, ground_truth} dicts
            
        Returns:
            Dict with aggregated metric scores
        """
        logger.info(f"Evaluating batch of {len(results)} queries...")
        
        all_scores = []
        
        for result in results:
            scores = self.evaluate_single(
                query=result["query"],
                answer=result["answer"],
                context=result["context"],
                ground_truth=result.get("ground_truth"),
            )
            all_scores.append(scores)
        
        # Aggregate scores
        aggregated = {}
        if all_scores:
            # Average across all samples
            for key in all_scores[0].keys():
                values = [s[key] for s in all_scores if key in s]
                if values:
                    aggregated[key] = sum(values) / len(values)
        
        logger.info(f"✓ Batch evaluation complete: {aggregated}")
        return aggregated
    
    def save_results(self, results: List[Dict], output_path: Optional[Path] = None):
        """Save evaluation results to CSV."""
        if output_path is None:
            output_path = DATA_DIR / "evaluation_results.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")


def evaluate_on_qna_dataset(orchestrator, qna_csv: str) -> Dict[str, float]:
    """
    Evaluate on QnA dataset.
    
    Args:
        orchestrator: RAG orchestrator
        qna_csv: Path to QnA CSV file
        
    Returns:
        Dict with evaluation metrics
    """
    logger.info(f"Loading QnA dataset from {qna_csv}...")
    
    try:
        df = pd.read_csv(qna_csv)
    except Exception as e:
        logger.error(f"Could not load QnA file: {e}")
        return {}
    
    evaluator = RAGEvaluator()
    results = []
    
    for idx, row in df.iterrows():
        try:
            query = row.get("question", "")
            ground_truth = row.get("answer", "")
            
            if not query:
                continue
            
            logger.info(f"[{idx+1}/{len(df)}] {query[:50]}...")
            
            # Run RAG pipeline
            state = orchestrator.invoke(query)
            
            # Collect result
            result = {
                "query": query,
                "answer": state.get("answer", ""),
                "context": state.get("context", ""),
                "ground_truth": ground_truth,
                "num_rag_chunks": len(state.get("rag_results", [])),
                "num_web_results": len(state.get("web_results", [])),
            }
            
            # Evaluate
            metrics = evaluator.evaluate_single(
                query=query,
                answer=result["answer"],
                context=result["context"],
                ground_truth=ground_truth,
            )
            result.update(metrics)
            results.append(result)
        
        except Exception as e:
            logger.error(f"Error processing query {idx}: {e}")
            continue
    
    # Save results
    evaluator.save_results(results)
    
    # Compute aggregate metrics
    aggregate = evaluator.evaluate_batch(results)
    
    return aggregate
