"""
Evaluation Module for Information Retrieval Search Engine
=========================================================
Computes standard IR evaluation metrics including:
- Precision, Recall, F1 Score
- Mean Average Precision (MAP)
- Precision at K (P@K)
- Normalized Discounted Cumulative Gain (NDCG)

Author: Tav
Course: Information Retrieval Final Project
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import os
import re


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    precision_at_k: Dict[int, float]
    ndcg: float
    num_relevant: int
    num_retrieved: int
    num_relevant_retrieved: int
    
    def __repr__(self):
        return (f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, "
                f"F1: {self.f1_score:.4f}, AP: {self.average_precision:.4f}")


class Evaluator:
    """
    Evaluates search engine performance against relevance judgments.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.relevance_judgments: Dict[str, Set[str]] = {}  # query_id -> set of relevant doc_ids
        self.queries: Dict[str, str] = {}  # query_id -> query text
    
    def load_relevance_judgments(self, filepath: str, format_type: str = "cacm"):
        """
        Load relevance judgments from file.
        
        Supports formats:
        - cacm: CACM format (query_id doc_id pairs)
        - trec: TREC qrels format (query_id 0 doc_id relevance)
        - simple: Simple format (query_id doc_id)
        
        Args:
            filepath: Path to relevance judgments file
            format_type: Format of the file
        """
        self.relevance_judgments = defaultdict(set)
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if format_type == "trec":
                    # TREC format: query_id 0 doc_id relevance
                    if len(parts) >= 4:
                        query_id = parts[0]
                        doc_id = parts[2]
                        relevance = int(parts[3])
                        if relevance > 0:
                            self.relevance_judgments[query_id].add(doc_id)
                            
                elif format_type == "cacm":
                    # CACM format: query_id doc_id (space or tab separated)
                    if len(parts) >= 2:
                        query_id = parts[0]
                        doc_id = parts[1]
                        self.relevance_judgments[query_id].add(doc_id)
                        
                elif format_type == "simple":
                    # Simple: query_id doc_id
                    if len(parts) >= 2:
                        query_id = parts[0]
                        doc_id = parts[1]
                        self.relevance_judgments[query_id].add(doc_id)
        
        print(f"Loaded relevance judgments for {len(self.relevance_judgments)} queries")
    
    def load_queries(self, filepath: str, format_type: str = "cacm"):
        """
        Load queries from file.
        
        Args:
            filepath: Path to queries file
            format_type: Format of the file
        """
        self.queries = {}
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        

        bracket_pattern = re.findall(r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)', content, re.DOTALL)
        if bracket_pattern:
            for query_id, query_text in bracket_pattern:
                # Clean up the query text (remove extra whitespace/newlines)
                cleaned_text = ' '.join(query_text.split())
                self.queries[query_id] = cleaned_text
        # Check if it's the standard CACM format with .I markers
        elif '.I' in content:
            current_id = None
            current_text = []
            in_query = False
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('.I'):
                    if current_id is not None and current_text:
                        self.queries[current_id] = ' '.join(current_text)
                    current_id = line.split()[1] if len(line.split()) > 1 else None
                    current_text = []
                    in_query = False
                elif line.startswith('.W'):
                    in_query = True
                elif line.startswith('.'):
                    in_query = False
                elif in_query and line:
                    current_text.append(line)
            
            # Don't forget last query
            if current_id is not None and current_text:
                self.queries[current_id] = ' '.join(current_text)
        else:
            # Simple format: one query per line, or numbered
            lines = content.strip().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    # Check if line starts with number
                    parts = line.split(None, 1)
                    if parts[0].isdigit() and len(parts) > 1:
                        self.queries[parts[0]] = parts[1]
                    else:
                        self.queries[str(i + 1)] = line
        
        print(f"Loaded {len(self.queries)} queries")
    
    def evaluate_single_query(self, 
                              query_id: str, 
                              retrieved_docs: List[str],
                              k_values: List[int] = [5, 10, 20]) -> Optional[EvaluationResults]:
        """
        Evaluate results for a single query.
        
        Args:
            query_id: Query identifier
            retrieved_docs: Ordered list of retrieved document IDs
            k_values: Values of K for P@K calculation
            
        Returns:
            EvaluationResults object or None if no judgments exist
        """
        if query_id not in self.relevance_judgments:
            return None
        
        relevant_docs = self.relevance_judgments[query_id]
        
        if not relevant_docs:
            return None
        
        # Basic counts
        num_relevant = len(relevant_docs)
        num_retrieved = len(retrieved_docs)
        relevant_retrieved = set(retrieved_docs) & relevant_docs
        num_relevant_retrieved = len(relevant_retrieved)
        
        # Precision and Recall
        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0
        recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Average Precision
        ap = self._calculate_average_precision(retrieved_docs, relevant_docs)
        
        # Precision at K
        p_at_k = {}
        for k in k_values:
            p_at_k[k] = self._precision_at_k(retrieved_docs, relevant_docs, k)
        
        # NDCG
        ndcg = self._calculate_ndcg(retrieved_docs, relevant_docs, num_retrieved)
        
        return EvaluationResults(
            precision=precision,
            recall=recall,
            f1_score=f1,
            average_precision=ap,
            precision_at_k=p_at_k,
            ndcg=ndcg,
            num_relevant=num_relevant,
            num_retrieved=num_retrieved,
            num_relevant_retrieved=num_relevant_retrieved
        )
    
    def _calculate_average_precision(self, 
                                     retrieved: List[str], 
                                     relevant: Set[str]) -> float:
        """Calculate Average Precision for a ranked list."""
        if not relevant:
            return 0.0
        
        num_relevant_seen = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant)
    
    def _precision_at_k(self, 
                        retrieved: List[str], 
                        relevant: Set[str], 
                        k: int) -> float:
        """Calculate Precision at K."""
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        
        return relevant_in_top_k / k
    
    def _calculate_ndcg(self, 
                        retrieved: List[str], 
                        relevant: Set[str], 
                        k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not relevant or not retrieved:
            return 0.0
        
        # DCG for retrieved results (binary relevance)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed
        
        # Ideal DCG (all relevant docs at top)
        idcg = 0.0
        num_relevant = min(len(relevant), k)
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_all_queries(self, 
                             results: Dict[str, List[str]],
                             k_values: List[int] = [5, 10, 20]) -> Dict[str, any]:
        """
        Evaluate results for all queries and compute aggregate metrics.
        
        Args:
            results: Dictionary mapping query_id to list of retrieved doc_ids
            k_values: Values of K for P@K calculation
            
        Returns:
            Dictionary with per-query and aggregate metrics
        """
        all_results = {}
        
        # Per-query evaluation
        query_metrics = {}
        for query_id, retrieved_docs in results.items():
            eval_result = self.evaluate_single_query(query_id, retrieved_docs, k_values)
            if eval_result is not None:
                query_metrics[query_id] = eval_result
        
        all_results['per_query'] = query_metrics
        
        # Aggregate metrics
        if query_metrics:
            num_queries = len(query_metrics)
            
            # Mean Average Precision (MAP)
            map_score = sum(r.average_precision for r in query_metrics.values()) / num_queries
            
            # Mean Precision
            mean_precision = sum(r.precision for r in query_metrics.values()) / num_queries
            
            # Mean Recall
            mean_recall = sum(r.recall for r in query_metrics.values()) / num_queries
            
            # Mean F1
            mean_f1 = sum(r.f1_score for r in query_metrics.values()) / num_queries
            
            # Mean P@K
            mean_p_at_k = {}
            for k in k_values:
                mean_p_at_k[k] = sum(r.precision_at_k[k] for r in query_metrics.values()) / num_queries
            
            # Mean NDCG
            mean_ndcg = sum(r.ndcg for r in query_metrics.values()) / num_queries
            
            all_results['aggregate'] = {
                'num_queries_evaluated': num_queries,
                'MAP': map_score,
                'mean_precision': mean_precision,
                'mean_recall': mean_recall,
                'mean_f1': mean_f1,
                'mean_p_at_k': mean_p_at_k,
                'mean_ndcg': mean_ndcg
            }
        else:
            all_results['aggregate'] = None
        
        return all_results
    
    def print_evaluation_report(self, evaluation: Dict[str, any]):
        """Print a formatted evaluation report."""
        print("\n" + "="*70)
        print("SEARCH ENGINE EVALUATION REPORT")
        print("="*70)
        
        if evaluation.get('aggregate'):
            agg = evaluation['aggregate']
            
            print(f"\nQUERIES EVALUATED: {agg['num_queries_evaluated']}")
            print("\n--- AGGREGATE METRICS ---")
            print(f"  Mean Average Precision (MAP): {agg['MAP']:.4f}")
            print(f"  Mean Precision:               {agg['mean_precision']:.4f}")
            print(f"  Mean Recall:                  {agg['mean_recall']:.4f}")
            print(f"  Mean F1 Score:                {agg['mean_f1']:.4f}")
            print(f"  Mean NDCG:                    {agg['mean_ndcg']:.4f}")
            
            print("\n--- PRECISION AT K ---")
            for k, p_at_k in sorted(agg['mean_p_at_k'].items()):
                print(f"  P@{k}: {p_at_k:.4f}")
        
        # Per-query details (top 10)
        if evaluation.get('per_query'):
            print("\n--- PER-QUERY RESULTS (sample) ---")
            count = 0
            for query_id, result in sorted(evaluation['per_query'].items()):
                if count >= 10:
                    print(f"  ... and {len(evaluation['per_query']) - 10} more queries")
                    break
                print(f"  Query {query_id}: P={result.precision:.3f}, R={result.recall:.3f}, "
                      f"F1={result.f1_score:.3f}, AP={result.average_precision:.3f}")
                count += 1
        
        print("\n" + "="*70)
    
    def export_results_csv(self, evaluation: Dict[str, any], filepath: str):
        """Export evaluation results to CSV."""
        with open(filepath, 'w') as f:
            # Header
            f.write("query_id,precision,recall,f1,average_precision,ndcg,"
                    "num_relevant,num_retrieved,num_relevant_retrieved\n")
            
            if evaluation.get('per_query'):
                for query_id, result in sorted(evaluation['per_query'].items()):
                    f.write(f"{query_id},{result.precision:.4f},{result.recall:.4f},"
                            f"{result.f1_score:.4f},{result.average_precision:.4f},"
                            f"{result.ndcg:.4f},{result.num_relevant},"
                            f"{result.num_retrieved},{result.num_relevant_retrieved}\n")
            
            # Aggregate row
            if evaluation.get('aggregate'):
                agg = evaluation['aggregate']
                f.write(f"AGGREGATE,{agg['mean_precision']:.4f},{agg['mean_recall']:.4f},"
                        f"{agg['mean_f1']:.4f},{agg['MAP']:.4f},{agg['mean_ndcg']:.4f},,,\n")
        
        print(f"Results exported to {filepath}")


def compare_models(evaluator: Evaluator, 
                   results_dict: Dict[str, Dict[str, List[str]]],
                   k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict]:
    """
    Compare multiple retrieval models/configurations.
    
    Args:
        evaluator: Configured Evaluator instance
        results_dict: Dict mapping model_name -> {query_id -> retrieved_docs}
        k_values: K values for P@K
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for model_name, results in results_dict.items():
        evaluation = evaluator.evaluate_all_queries(results, k_values)
        comparison[model_name] = evaluation.get('aggregate', {})
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'MAP':<10} {'P@5':<10} {'P@10':<10} {'F1':<10} {'NDCG':<10}")
    print("-"*80)
    
    for model_name, metrics in comparison.items():
        if metrics:
            p5 = metrics['mean_p_at_k'].get(5, 0)
            p10 = metrics['mean_p_at_k'].get(10, 0)
            print(f"{model_name:<20} {metrics['MAP']:<10.4f} {p5:<10.4f} "
                  f"{p10:<10.4f} {metrics['mean_f1']:<10.4f} {metrics['mean_ndcg']:<10.4f}")
    
    print("="*80)
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Demo with sample data
    evaluator = Evaluator()
    
    # Create sample relevance judgments
    evaluator.relevance_judgments = {
        "1": {"doc1", "doc3", "doc5", "doc7"},
        "2": {"doc2", "doc4", "doc6"},
        "3": {"doc1", "doc2", "doc8", "doc9", "doc10"}
    }
    
    # Sample search results
    sample_results = {
        "1": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"],
        "2": ["doc4", "doc2", "doc1", "doc6", "doc3", "doc5", "doc7", "doc8", "doc9", "doc10"],
        "3": ["doc1", "doc8", "doc3", "doc2", "doc10", "doc4", "doc9", "doc5", "doc6", "doc7"]
    }
    
    # Evaluate
    evaluation = evaluator.evaluate_all_queries(sample_results)
    evaluator.print_evaluation_report(evaluation)