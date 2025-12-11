"""
Search Engine Runner - Final Project
=====================================
Main script to run indexing, searching, and evaluation.

Usage:
    python run_evaluation.py --data-dir <path> --queries <path> --qrels <path>
    
Author: Tav
Course: Information Retrieval Final Project
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

# Import our modules
import indexer
from evaluator import Evaluator, compare_models


def run_full_evaluation(data_dir: str, 
                        queries_file: str = None, 
                        qrels_file: str = None,
                        top_k: int = 100) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        data_dir: Directory containing documents to index
        queries_file: Path to queries file (optional, for evaluation)
        qrels_file: Path to relevance judgments file (optional, for evaluation)
        top_k: Number of results to retrieve per query
        
    Returns:
        Dictionary containing evaluation results and timing info
    """
    results = {
        'timing': {},
        'index_stats': {},
        'evaluation': None
    }
    
    print("\n" + "="*70)
    print("SEARCH ENGINE EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Create index
    print("\n[1/4] Creating database and indexing documents...")
    start_time = time.time()
    
    indexer.create_db()
    indexer.index_dir(data_dir)
    
    index_time = time.time() - start_time
    results['timing']['indexing'] = index_time
    print(f"      Indexing completed in {index_time:.2f} seconds")
    
    # Get index statistics
    import sqlite3
    conn = sqlite3.connect('index.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM indexed_files')
    num_docs = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT term) FROM inverted_index')
    vocab_size = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM inverted_index')
    num_postings = cursor.fetchone()[0]
    
    conn.close()
    
    results['index_stats'] = {
        'num_documents': num_docs,
        'vocabulary_size': vocab_size,
        'num_postings': num_postings
    }
    
    print(f"      Documents indexed: {num_docs}")
    print(f"      Vocabulary size: {vocab_size}")
    print(f"      Total postings: {num_postings}")
    
    # Step 2: Load queries if provided
    if queries_file and os.path.exists(queries_file):
        print("\n[2/4] Loading queries...")
        evaluator = Evaluator()
        evaluator.load_queries(queries_file)
        
        queries = evaluator.queries
        print(f"      Loaded {len(queries)} queries")
    else:
        print("\n[2/4] No queries file provided - using sample queries")
        queries = {
            "1": "information retrieval",
            "2": "computer programming",
            "3": "data structures algorithms",
            "4": "machine learning",
            "5": "database systems"
        }
        evaluator = Evaluator()
        evaluator.queries = queries
    
    # Step 3: Run searches
    print("\n[3/4] Running searches...")
    start_time = time.time()
    
    search_results = {}
    for query_id, query_text in queries.items():
        ranked_docs = indexer.search(query_text)[:top_k]
        
        # Extract just the doc identifiers (filenames or doc IDs)
        # Normalize to just the filename for matching with qrels
        doc_ids = []
        for doc_path in ranked_docs:
            # Extract document ID from path
            doc_name = os.path.basename(doc_path)
            # Remove extension if needed
            doc_id = os.path.splitext(doc_name)[0]
            doc_ids.append(doc_id)
        
        search_results[query_id] = doc_ids
    
    search_time = time.time() - start_time
    results['timing']['searching'] = search_time
    print(f"      Searches completed in {search_time:.2f} seconds")
    print(f"      Average query time: {search_time/len(queries)*1000:.2f} ms")
    
    # Step 4: Evaluate if qrels provided
    if qrels_file:
        print(f"\n[4/4] Evaluating results...")
        print(f"      Looking for qrels file: {qrels_file}")
        if os.path.exists(qrels_file):
            print(f"      Found qrels file!")
            evaluator.load_relevance_judgments(qrels_file, format_type="trec")
            
            evaluation = evaluator.evaluate_all_queries(search_results)
            results['evaluation'] = evaluation
            
            # Print report
            evaluator.print_evaluation_report(evaluation)
            
            # Export to CSV
            evaluator.export_results_csv(evaluation, 'evaluation_results.csv')
        else:
            print(f"      ERROR: qrels file not found at: {os.path.abspath(qrels_file)}")
    else:
        print("\n[4/4] No qrels file provided - skipping evaluation")
        print("      To run evaluation, provide relevance judgments with --qrels")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total indexing time:  {results['timing']['indexing']:.2f} s")
    print(f"  Total search time:    {results['timing']['searching']:.2f} s")
    print(f"  Documents indexed:    {results['index_stats']['num_documents']}")
    print(f"  Queries processed:    {len(queries)}")
    
    if results['evaluation'] and results['evaluation'].get('aggregate'):
        agg = results['evaluation']['aggregate']
        print(f"\n  EVALUATION METRICS:")
        print(f"    MAP:                {agg['MAP']:.4f}")
        print(f"    Mean F1:            {agg['mean_f1']:.4f}")
        print(f"    Mean Precision:     {agg['mean_precision']:.4f}")
        print(f"    Mean Recall:        {agg['mean_recall']:.4f}")
    
    print("="*70)
    
    return results


def interactive_demo():
    """Run an interactive search demo."""
    print("\n" + "="*50)
    print("INTERACTIVE SEARCH DEMO")
    print("="*50)
    print("Type a query to search, or 'quit' to exit\n")
    
    while True:
        try:
            query = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Run search
        start = time.time()
        results = indexer.search(query)
        elapsed = time.time() - start
        
        print(f"\nFound {len(results)} results in {elapsed*1000:.2f} ms\n")
        
        # Display top 10
        for i, doc in enumerate(results[:10], 1):
            title = indexer.get_document_title(doc)
            print(f"  {i}. {title}")
            print(f"     {doc}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Information Retrieval Search Engine - Final Project"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        default="../data/cacm/docs",
        help="Directory containing documents to index"
    )
    
    parser.add_argument(
        "--queries", "-q",
        help="Path to queries file"
    )
    
    parser.add_argument(
        "--qrels", "-r",
        help="Path to relevance judgments (qrels) file"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query (default: 100)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive search mode"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with sample data for demonstration"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found")
        print("Please provide a valid --data-dir path")
        sys.exit(1)
    
    # Run evaluation
    results = run_full_evaluation(
        data_dir=args.data_dir,
        queries_file=args.queries,
        qrels_file=args.qrels,
        top_k=args.top_k
    )
    
    # Optional interactive mode
    if args.interactive:
        interactive_demo()
    
    return results


if __name__ == "__main__":
    main()