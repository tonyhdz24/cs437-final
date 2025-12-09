"""
Retriever Module for Information Retrieval Search Engine
========================================================
Implements multiple retrieval models including TF-IDF Vector Space Model
and BM25. Supports boolean queries and pseudo-relevance feedback.

Author: Tav
Course: Information Retrieval Final Project
"""

import math
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from indexer import InvertedIndex, TextPreprocessor


class RankingModel(Enum):
    """Available ranking models."""
    TFIDF = "tfidf"
    BM25 = "bm25"


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    score: float
    rank: int
    title: str = ""
    snippet: str = ""
    
    def __repr__(self):
        return f"SearchResult(rank={self.rank}, doc_id='{self.doc_id}', score={self.score:.4f})"


class Retriever:
    """
    Main retrieval class supporting multiple ranking models.
    """
    
    def __init__(self, index: InvertedIndex):
        """
        Initialize the retriever with an inverted index.
        
        Args:
            index: Built InvertedIndex instance
        """
        self.index = index
        self.preprocessor = index.preprocessor
        
        # BM25 parameters (tunable)
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        self.k3 = 1.5  # Query term frequency parameter (for long queries)
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               model: RankingModel = RankingModel.BM25) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            model: Ranking model to use (TFIDF or BM25)
            
        Returns:
            List of SearchResult objects, sorted by score descending
        """
        # Preprocess query
        query_terms = self.preprocessor.preprocess(query)
        
        if not query_terms:
            return []
        
        # Get scores based on model
        if model == RankingModel.TFIDF:
            scores = self._tfidf_score(query_terms)
        elif model == RankingModel.BM25:
            scores = self._bm25_score(query_terms)
        else:
            raise ValueError(f"Unknown ranking model: {model}")
        
        # Get top-k results
        top_docs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        
        # Build result objects
        results = []
        for rank, (doc_id, score) in enumerate(top_docs, 1):
            metadata = self.index.doc_metadata.get(doc_id, {})
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', '')
            )
            results.append(result)
        
        return results
    
    def _tfidf_score(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF cosine similarity scores.
        
        Uses the vector space model with log-normalized TF and IDF weighting.
        """
        scores = defaultdict(float)
        
        # Build query vector
        query_tf = defaultdict(int)
        for term in query_terms:
            query_tf[term] += 1
        
        # Calculate query vector weights
        query_weights = {}
        query_norm = 0.0
        
        for term, tf in query_tf.items():
            if term in self.index.vocabulary:
                # Log-normalized TF for query
                tf_weight = 1 + math.log(tf) if tf > 0 else 0
                idf = self.index.get_idf(term)
                weight = tf_weight * idf
                query_weights[term] = weight
                query_norm += weight ** 2
        
        query_norm = math.sqrt(query_norm) if query_norm > 0 else 1
        
        # Calculate document scores
        doc_norms = {}
        
        for term in query_terms:
            if term not in self.index.vocabulary:
                continue
            
            query_weight = query_weights.get(term, 0)
            idf = self.index.get_idf(term)
            
            for posting in self.index.get_postings(term):
                doc_id = posting.doc_id
                
                # Log-normalized TF for document
                tf_weight = 1 + math.log(posting.term_frequency) if posting.term_frequency > 0 else 0
                doc_weight = tf_weight * idf
                
                # Accumulate dot product
                scores[doc_id] += query_weight * doc_weight
                
                # Track document norms
                if doc_id not in doc_norms:
                    doc_norms[doc_id] = 0.0
                doc_norms[doc_id] += doc_weight ** 2
        
        # Normalize scores (cosine similarity)
        for doc_id in scores:
            doc_norm = math.sqrt(doc_norms.get(doc_id, 1))
            scores[doc_id] /= (query_norm * doc_norm) if doc_norm > 0 else 1
        
        return dict(scores)
    
    def _bm25_score(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate BM25 scores.
        
        BM25 is typically more effective than TF-IDF for ranked retrieval.
        """
        scores = defaultdict(float)
        
        # Query term frequencies
        query_tf = defaultdict(int)
        for term in query_terms:
            query_tf[term] += 1
        
        avg_dl = self.index.avg_doc_length
        N = self.index.num_docs
        
        for term in query_terms:
            if term not in self.index.vocabulary:
                continue
            
            # Document frequency
            df = self.index.doc_frequencies[term]
            
            # IDF component (BM25 version)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # Query term frequency component
            qtf = query_tf[term]
            query_component = ((self.k3 + 1) * qtf) / (self.k3 + qtf)
            
            for posting in self.index.get_postings(term):
                doc_id = posting.doc_id
                tf = posting.term_frequency
                dl = self.index.doc_lengths.get(doc_id, avg_dl)
                
                # BM25 term frequency component
                tf_component = ((self.k1 + 1) * tf) / (self.k1 * (1 - self.b + self.b * (dl / avg_dl)) + tf)
                
                # Combine components
                scores[doc_id] += idf * tf_component * query_component
        
        return dict(scores)
    
    def boolean_search(self, 
                       query: str, 
                       top_k: int = 10,
                       ranking_model: RankingModel = RankingModel.BM25) -> List[SearchResult]:
        """
        Perform boolean search with AND, OR, NOT operators.
        
        Query syntax:
            - word1 AND word2: Both terms must appear
            - word1 OR word2: Either term must appear  
            - NOT word: Exclude documents with term
            - Parentheses for grouping (limited support)
        
        Args:
            query: Boolean query string
            top_k: Number of results to return
            ranking_model: Model to rank matching documents
            
        Returns:
            List of SearchResult objects
        """
        # Parse boolean query
        query_upper = query.upper()
        
        # Simple parsing - split on operators
        if ' AND ' in query_upper:
            parts = query.split(' AND ')
            parts = [query.split(' and ') for query in parts]
            parts = [p.strip() for sublist in [[p] if isinstance(p, str) else p for p in parts] for p in (sublist if isinstance(sublist, list) else [sublist])]
            # Handle nested splits
            flat_parts = []
            for p in query.upper().split(' AND '):
                flat_parts.append(p.strip())
            
            result_docs = None
            for part in flat_parts:
                part_lower = part.lower()
                terms = self.preprocessor.preprocess(part_lower)
                part_docs = self._get_docs_for_terms(terms)
                
                if result_docs is None:
                    result_docs = part_docs
                else:
                    result_docs = result_docs.intersection(part_docs)
            
            candidate_docs = result_docs or set()
            
        elif ' OR ' in query_upper:
            flat_parts = []
            for p in query.upper().split(' OR '):
                flat_parts.append(p.strip())
            
            result_docs = set()
            for part in flat_parts:
                part_lower = part.lower()
                terms = self.preprocessor.preprocess(part_lower)
                part_docs = self._get_docs_for_terms(terms)
                result_docs = result_docs.union(part_docs)
            
            candidate_docs = result_docs
            
        elif query_upper.startswith('NOT '):
            # NOT query - exclude these docs
            exclude_query = query[4:]
            terms = self.preprocessor.preprocess(exclude_query.lower())
            exclude_docs = self._get_docs_for_terms(terms)
            candidate_docs = set(self.index.doc_lengths.keys()) - exclude_docs
            
        else:
            # No boolean operators - standard search
            return self.search(query, top_k, ranking_model)
        
        if not candidate_docs:
            return []
        
        # Rank the candidate documents
        query_terms = self.preprocessor.preprocess(query.lower())
        # Remove operator words
        query_terms = [t for t in query_terms if t not in {'and', 'or', 'not'}]
        
        if ranking_model == RankingModel.BM25:
            all_scores = self._bm25_score(query_terms)
        else:
            all_scores = self._tfidf_score(query_terms)
        
        # Filter to candidate docs
        filtered_scores = {doc_id: score for doc_id, score in all_scores.items() 
                         if doc_id in candidate_docs}
        
        # For docs with no score, give small score so they appear
        for doc_id in candidate_docs:
            if doc_id not in filtered_scores:
                filtered_scores[doc_id] = 0.001
        
        # Get top-k
        top_docs = heapq.nlargest(top_k, filtered_scores.items(), key=lambda x: x[1])
        
        results = []
        for rank, (doc_id, score) in enumerate(top_docs, 1):
            metadata = self.index.doc_metadata.get(doc_id, {})
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', '')
            )
            results.append(result)
        
        return results
    
    def _get_docs_for_terms(self, terms: List[str]) -> Set[str]:
        """Get all documents containing any of the given terms."""
        docs = set()
        for term in terms:
            for posting in self.index.get_postings(term):
                docs.add(posting.doc_id)
        return docs
    
    def search_with_feedback(self,
                             query: str,
                             top_k: int = 10,
                             feedback_docs: int = 3,
                             expansion_terms: int = 5,
                             alpha: float = 1.0,
                             beta: float = 0.8,
                             model: RankingModel = RankingModel.BM25) -> Tuple[List[SearchResult], List[str]]:
        """
        Pseudo-relevance feedback (query expansion).
        
        Assumes top documents from initial search are relevant and
        expands query with terms from those documents.
        
        Args:
            query: Original query
            top_k: Number of final results
            feedback_docs: Number of top docs to use for feedback
            expansion_terms: Number of terms to add to query
            alpha: Weight for original query terms
            beta: Weight for expansion terms
            model: Ranking model
            
        Returns:
            Tuple of (results, expanded_terms)
        """
        # Initial search
        initial_results = self.search(query, top_k=feedback_docs, model=model)
        
        if not initial_results:
            return [], []
        
        # Get original query terms
        original_terms = self.preprocessor.preprocess(query)
        original_set = set(original_terms)
        
        # Collect term scores from feedback documents
        term_scores = defaultdict(float)
        
        for result in initial_results:
            doc_id = result.doc_id
            
            # Get all terms in this document with their TF-IDF weights
            for term in self.index.vocabulary:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf > 0 and term not in original_set:
                    idf = self.index.get_idf(term)
                    tf_weight = 1 + math.log(tf)
                    term_scores[term] += tf_weight * idf
        
        # Select top expansion terms
        top_expansion = heapq.nlargest(expansion_terms, term_scores.items(), key=lambda x: x[1])
        expansion_term_list = [term for term, _ in top_expansion]
        
        # Build expanded query with weights
        expanded_query_terms = original_terms.copy()
        
        # Add expansion terms (could be added multiple times based on score)
        for term in expansion_term_list:
            expanded_query_terms.append(term)
        
        # Re-run search with expanded query
        if model == RankingModel.BM25:
            scores = self._bm25_score(expanded_query_terms)
        else:
            scores = self._tfidf_score(expanded_query_terms)
        
        # Apply Rocchio-style weighting (simplified)
        # Original query contribution
        original_scores = self._bm25_score(original_terms) if model == RankingModel.BM25 else self._tfidf_score(original_terms)
        
        # Combine scores
        final_scores = {}
        all_docs = set(scores.keys()) | set(original_scores.keys())
        
        for doc_id in all_docs:
            original_score = original_scores.get(doc_id, 0)
            expanded_score = scores.get(doc_id, 0)
            final_scores[doc_id] = alpha * original_score + beta * expanded_score
        
        # Get top-k
        top_docs = heapq.nlargest(top_k, final_scores.items(), key=lambda x: x[1])
        
        results = []
        for rank, (doc_id, score) in enumerate(top_docs, 1):
            metadata = self.index.doc_metadata.get(doc_id, {})
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', '')
            )
            results.append(result)
        
        return results, expansion_term_list
    
    def phrase_search(self, 
                      phrase: str, 
                      top_k: int = 10,
                      model: RankingModel = RankingModel.BM25) -> List[SearchResult]:
        """
        Search for an exact phrase using positional information.
        
        Args:
            phrase: Exact phrase to search for
            top_k: Number of results
            model: Ranking model for scoring
            
        Returns:
            List of SearchResult objects
        """
        terms = self.preprocessor.preprocess(phrase)
        
        if len(terms) < 2:
            # Single term - do regular search
            return self.search(phrase, top_k, model)
        
        # Find documents containing all terms
        candidate_docs = None
        
        for term in terms:
            term_docs = set()
            for posting in self.index.get_postings(term):
                term_docs.add(posting.doc_id)
            
            if candidate_docs is None:
                candidate_docs = term_docs
            else:
                candidate_docs = candidate_docs.intersection(term_docs)
        
        if not candidate_docs:
            return []
        
        # Check positional constraints
        phrase_docs = set()
        
        for doc_id in candidate_docs:
            # Get positions for each term
            term_positions = []
            for term in terms:
                positions = []
                for posting in self.index.get_postings(term):
                    if posting.doc_id == doc_id:
                        positions = posting.positions
                        break
                term_positions.append(set(positions))
            
            # Check if terms appear consecutively
            if self._check_phrase_positions(term_positions):
                phrase_docs.add(doc_id)
        
        if not phrase_docs:
            return []
        
        # Score and rank
        if model == RankingModel.BM25:
            all_scores = self._bm25_score(terms)
        else:
            all_scores = self._tfidf_score(terms)
        
        filtered_scores = {doc_id: all_scores.get(doc_id, 0.001) 
                         for doc_id in phrase_docs}
        
        top_docs = heapq.nlargest(top_k, filtered_scores.items(), key=lambda x: x[1])
        
        results = []
        for rank, (doc_id, score) in enumerate(top_docs, 1):
            metadata = self.index.doc_metadata.get(doc_id, {})
            result = SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', '')
            )
            results.append(result)
        
        return results
    
    def _check_phrase_positions(self, term_positions: List[Set[int]]) -> bool:
        """Check if positions allow consecutive occurrence."""
        if not term_positions or not term_positions[0]:
            return False
        
        # For each starting position of first term
        for start_pos in term_positions[0]:
            found = True
            for i, positions in enumerate(term_positions[1:], 1):
                if (start_pos + i) not in positions:
                    found = False
                    break
            if found:
                return True
        
        return False
    
    def set_bm25_parameters(self, k1: float = 1.5, b: float = 0.75, k3: float = 1.5):
        """
        Set BM25 tuning parameters.
        
        Args:
            k1: Term frequency saturation (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
            k3: Query term frequency parameter
        """
        self.k1 = k1
        self.b = b
        self.k3 = k3


class BatchRetriever:
    """
    Handles batch query processing for evaluation.
    """
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    def process_queries(self,
                        queries: Dict[str, str],
                        top_k: int = 100,
                        model: RankingModel = RankingModel.BM25) -> Dict[str, List[SearchResult]]:
        """
        Process multiple queries.
        
        Args:
            queries: Dictionary mapping query_id to query text
            top_k: Number of results per query
            model: Ranking model
            
        Returns:
            Dictionary mapping query_id to list of results
        """
        results = {}
        
        for query_id, query_text in queries.items():
            results[query_id] = self.retriever.search(query_text, top_k, model)
        
        return results
    
    def export_trec_format(self,
                           results: Dict[str, List[SearchResult]],
                           run_name: str = "my_run",
                           output_file: str = "results.txt") -> None:
        """
        Export results in TREC format for evaluation with trec_eval.
        
        Format: query_id Q0 doc_id rank score run_name
        """
        with open(output_file, 'w') as f:
            for query_id, query_results in results.items():
                for result in query_results:
                    f.write(f"{query_id} Q0 {result.doc_id} {result.rank} {result.score:.6f} {run_name}\n")
        
        print(f"Results exported to {output_file}")


def interactive_search(index_path: str):
    """
    Run an interactive search session.
    
    Args:
        index_path: Path to saved index file
    """
    print("Loading index...")
    index = InvertedIndex.load(index_path)
    retriever = Retriever(index)
    
    print("\n" + "="*60)
    print("Interactive Search Engine")
    print("="*60)
    print("Commands:")
    print("  <query>           - Standard BM25 search")
    print("  /tfidf <query>    - TF-IDF search")
    print("  /phrase <phrase>  - Exact phrase search")
    print("  /bool <query>     - Boolean search (AND, OR, NOT)")
    print("  /expand <query>   - Search with query expansion")
    print("  /stats            - Show index statistics")
    print("  /quit             - Exit")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("Search> ").strip()
        except EOFError:
            break
        
        if not query:
            continue
        
        if query.lower() == '/quit':
            print("Goodbye!")
            break
        
        if query.lower() == '/stats':
            stats = index.get_stats()
            print("\nIndex Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            continue
        
        # Parse command
        if query.startswith('/tfidf '):
            results = retriever.search(query[7:], top_k=10, model=RankingModel.TFIDF)
            model_name = "TF-IDF"
        elif query.startswith('/phrase '):
            results = retriever.phrase_search(query[8:], top_k=10)
            model_name = "Phrase"
        elif query.startswith('/bool '):
            results = retriever.boolean_search(query[6:], top_k=10)
            model_name = "Boolean"
        elif query.startswith('/expand '):
            results, expansion = retriever.search_with_feedback(query[8:], top_k=10)
            model_name = "Expanded"
            if expansion:
                print(f"  Query expanded with: {', '.join(expansion)}")
        else:
            results = retriever.search(query, top_k=10, model=RankingModel.BM25)
            model_name = "BM25"
        
        # Display results
        print(f"\n{model_name} Results ({len(results)} found):")
        print("-" * 50)
        
        if not results:
            print("  No results found.")
        else:
            for r in results:
                title_display = r.title[:50] + "..." if len(r.title) > 50 else r.title
                print(f"  {r.rank}. [{r.doc_id}] {title_display}")
                print(f"     Score: {r.score:.4f}")
        
        print()


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search documents using inverted index")
    parser.add_argument("index_file", help="Path to saved index file")
    parser.add_argument("-q", "--query", help="Single query to execute")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("-m", "--model", choices=['bm25', 'tfidf'], default='bm25', help="Ranking model")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_search(args.index_file)
    elif args.query:
        # Single query mode
        index = InvertedIndex.load(args.index_file)
        retriever = Retriever(index)
        
        model = RankingModel.BM25 if args.model == 'bm25' else RankingModel.TFIDF
        results = retriever.search(args.query, args.top_k, model)
        
        print(f"\nResults for: {args.query}")
        print("-" * 50)
        for r in results:
            print(f"{r.rank}. [{r.doc_id}] Score: {r.score:.4f}")
            if r.title:
                print(f"   Title: {r.title}")
    else:
        print("Please provide a query with -q or use -i for interactive mode")