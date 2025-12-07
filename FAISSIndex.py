import faiss
import numpy as np

# Corrected FAISSIndex class definition
class FAISSIndex:
    def __init__(self, embeddings):
        # Normalize embeddings to unit length for cosine similarity with IndexFlatIP
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.dimension = embeddings.shape[1] # Store the dimension as an instance attribute
        self.index = faiss.IndexFlatIP(self.dimension) # Initialize FAISS index with the stored dimension
        self.index.add(self.embeddings) # Add the embeddings to the index

    def search(self, query_embedding, top_k):
        # Normalize query embedding
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
        # Reshape for faiss search (faiss expects a 2D array, even for a single query)
        query_embedding_reshaped = query_embedding_normalized.reshape(1, -1)
        distances, indices = self.index.search(query_embedding_reshaped, top_k)
        return indices[0], distances[0]

# The original import statement 'from FAISSIndex import FAISSIndex' is not strictly necessary
# if the class is defined directly here, as this definition will take precedence.
# However, it doesn't cause an issue if left in.
