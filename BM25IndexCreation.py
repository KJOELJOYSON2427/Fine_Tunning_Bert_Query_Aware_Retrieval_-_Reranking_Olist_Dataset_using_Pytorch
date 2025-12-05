from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd


class BM25SearchEngine:
    def __init__(self, dataframe, tokenize, text_column="review_text"):
        """
        Initialize BM25 search engine with a pandas DataFrame.
        
        Args:
            dataframe (pd.DataFrame): Your dataset
            tokenize (callable): Your tokenizer function
            text_column (str): Column name containing text to search
        """
        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        self.tokenize = tokenize

        # Clean & tokenize documents
        print("Tokenizing documents...")
        self.tokenized_docs = [
            self.tokenize(text) for text in self.df[text_column].astype(str)
        ]

        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"BM25 Index created for {len(self.tokenized_docs)} documents!")

        self.detect_language()

    def detect_language(self):
        words = ['bom', 'otimo', 'produto', 'entrega', 'qualidade']
        sample_texts = self.df['review_text'].head(10).tolist()

        print("\nSample reviews (checking language):")
        for i, text in enumerate(sample_texts[:3], 1):
            print(f"{i}. {text[:80]}...")

        # Check if Portuguese
        is_portuguese = any(word in ''.join(sample_texts).lower() for word in words)
        if is_portuguese:
            print("\nDetected: Reviews are in Portuguese")
            print("Recommendation: Use Portuguese queries for BM25, or rely more on semantic search")

    def search(self, query, top_n=5, show_only_nonzero=True, verbose=False):
        """
        Search the corpus using BM25.
        
        Args:
            query (str): Search query string
            top_n (int): Number of results
            show_only_nonzero (bool): Filter out zero-score results
            verbose (bool): Print human-readable output
            
        Returns:
            pd.DataFrame: Top matching rows with BM25 scores
        """

        # Tokenize query
        tokenized_query = self.tokenize(query)

        # Compute BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_n highest scoring indices
        top_idx = np.argsort(scores)[::-1][:top_n]

        # Create results DataFrame
        results = self.df.iloc[top_idx].copy()
        results["bm25_score"] = np.array(scores)[top_idx]

        # Filter out zero-score documents (optional)
        if show_only_nonzero:
            results = results[results["bm25_score"] > 0]

        # If verbose, print results nicely
        if verbose:
            print(f"\nQuery: '{query}'")
            print("=" * 70)

            if results.empty:
                print("No matching results found (all BM25 scores = 0)")
                return results

            for i, row in results.iterrows():
                print(f"\nScore: {row['bm25_score']:.3f}")
                print(f"Review: {row[self.text_column][:120]}...")
            print("\n")

        return results