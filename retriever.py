
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class QueryEngine:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """Initialize query engine with embedding model"""
        self.model = SentenceTransformer(model_name)
        self.vector_store = None
        self.chunk_map = None

    def load_vector_store(self, index_path: str, chunk_map_path: str):
        """Load vector store from saved files"""
        # Load FAISS index
        self.vector_store = faiss.read_index(index_path)

        # Load chunk mapping (metadata about the chunks)
        with open(chunk_map_path, 'rb') as f:
            self.chunk_map = pickle.load(f)

    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        """Process query and return relevant documents"""
        # Generate query embedding
        query_vector = self.model.encode([query_text])[0]

        # Search vector store
        distances, indices = self.vector_store.search(
            query_vector.reshape(1, -1).astype('float32'),
            k
        )

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_map):
                result = self.chunk_map[idx].copy()
                result['distance'] = float(distances[0][i])
                results.append(result)

        return results

    def retrieve_context(self, query: str, top_k=10) -> str:
        """
        Retrieve a concatenated string of top-k relevant document snippets for a query.

        Args:
            query (str): The input query text.
            top_k (int): Number of top relevant documents to retrieve.

        Returns:
            str: A single concatenated string of relevant document snippets.
        """
        # Step 1: Query for top-k relevant documents using the existing `query` method
        results = self.query(query, top_k)

        # Step 2: Concatenate snippets from results
        snippets = []
        for result in results:
            # Ensure we use 'text' for the snippet
            snippet = result.get('text', '')
            snippets.append(snippet)

        # Concatenate snippets into a single context string
        context = " ".join(snippets)
        return context

    def print_results(self, results: List[Dict]):
        """Print search results in a readable format"""
        print("\nSearch Results:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Document: {result['document']}")
            print(f"Relevance Score: {1/(1 + result['distance']):.3f}")
            print(f"Text Snippet: {result['text'][:200]}...")
            print("-" * 80)


# Example usage and test queries
if __name__ == "__main__":
    # Initialize query engine
    query_engine = QueryEngine()

    # Load the vector store and chunk mapping
    query_engine.load_vector_store(
        'vector_store/faiss_index.idx', 'vector_store/chunk_map.pkl')

    # Define the query
    query = "what is the difference between google and tesla total revenue"

    # Retrieve concatenated context for the query
    context = query_engine.retrieve_context(query, top_k=10)

    # Print the concatenated context
    print("\nContext for LLM:")
    print("-" * 80)
    print(context)
    print("-" * 80)
