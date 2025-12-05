from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


class SemanticEmbeddingEngine:
    def __init__(self, dataframe, text_column="review_text", model_name="all-MiniLM-L6-v2", batch_size=32):
        """
        Creates a semantic embedding search engine using SentenceTransformer.
        
        Args:
            dataframe (pd.DataFrame): Dataset with text
            text_column (str): Column with text to encode
            model_name (str): SBERT model name
            batch_size (int): Batch size for encoding
        """
        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        self.batch_size = batch_size

        print(f"Loading semantic model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        print("Encoding documents into embeddings...")
        self.corpus_texts = self.df[text_column].astype(str).tolist()
        print(self.corpus_texts)


        self.corpus_embeddings = self.model.encode(
            self.corpus_texts,
            batch_size = self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"\nEmbeddings created!")
        print(f"Shape: {self.corpus_embeddings.shape}")
        
    

    def search(self, query, top_n=5):
        """
        Perform semantic search using cosine similarity.
        
        Args:
            query (str): Query string
            top_n (int): Number of results to return
        
        Returns:
            DataFrame containing the top results + similarity score
        """
        query_emb = self.model.encode(query, convert_to_numpy=True)
        scores = util.cos_sim(query_emb, self.corpus_embeddings)[0].cpu()


        # Sort best matches
        top_idx = np.argsort(scores)[::-1][:top_n]
        
        results = self.df.iloc[top_idx].copy()
        results["semantic_score"] = scores[top_idx]

        return results


