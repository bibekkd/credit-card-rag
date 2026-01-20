"""
Credit Card RAG - Pinecone Indexing
====================================

This module handles indexing of credit card embeddings into Pinecone vector database.

Features:
- Creates/connects to Pinecone index
- Uploads embeddings with metadata
- Batch processing for efficiency
- Progress tracking
- Index management (create, delete, stats)
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()


class PineconeIndexer:
    """
    Handles indexing of credit card embeddings into Pinecone.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        index_name: str = "credit-cards",
        dimension: int = 1024,
        metric: str = "cosine"
    ):
        """
        Initialize the Pinecone indexer.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            dimension: Dimension of embeddings (1024 for mistral-embed)
            metric: Distance metric (cosine, euclidean, or dotproduct)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key not found. Set PINECONE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        print(f"✓ Initialized Pinecone client")
        print(f"  Index name: {self.index_name}")
        print(f"  Dimension: {self.dimension}")
        print(f"  Metric: {self.metric}")
    
    def create_index(self, cloud: str = "aws", region: str = "us-east-1"):
        """
        Create a new Pinecone index if it doesn't exist.
        
        Args:
            cloud: Cloud provider (aws, gcp, or azure)
            region: Cloud region
        """
        # Check if index already exists
        existing_indexes = self.pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if self.index_name in index_names:
            print(f"✓ Index '{self.index_name}' already exists")
            return
        
        print(f"Creating new index '{self.index_name}'...")
        
        # Create serverless index
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        print(f"✓ Index '{self.index_name}' created successfully")
    
    def get_index(self):
        """
        Get the Pinecone index object.
        
        Returns:
            Pinecone Index object
        """
        return self.pc.Index(self.index_name)
    
    def index_embeddings(
        self, 
        embeddings_path: str = "./embeddings/credit_cards_embeddings.json",
        batch_size: int = 100,
        namespace: str = ""
    ):
        """
        Index embeddings into Pinecone.
        
        Args:
            embeddings_path: Path to embeddings JSON file
            batch_size: Number of vectors to upload per batch
            namespace: Pinecone namespace (optional, for data organization)
        """
        print(f"\n{'='*80}")
        print(f"INDEXING EMBEDDINGS TO PINECONE")
        print(f"{'='*80}\n")
        
        # Load embeddings
        print(f"Loading embeddings from: {embeddings_path}")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embedded_docs = json.load(f)
        
        print(f"✓ Loaded {len(embedded_docs)} embeddings")
        
        # Get index
        index = self.get_index()
        
        # Prepare vectors for upload
        vectors = []
        for doc in embedded_docs:
            # Prepare metadata (Pinecone has limitations on metadata)
            metadata = {
                "card_name": doc["metadata"].get("card_name", ""),
                "bank": doc["metadata"].get("bank", ""),
                "category": doc["metadata"].get("category", ""),
                "reward_type": doc["metadata"].get("reward_type", ""),
                "use_case": doc["metadata"].get("use_case", ""),
                "source": doc["metadata"].get("source", ""),
                "text": doc["text"][:1000]  # Limit text to 1000 chars for metadata
            }
            
            vectors.append({
                "id": doc["id"],
                "values": doc["embedding"],
                "metadata": metadata
            })
        
        # Upload in batches
        print(f"\nUploading {len(vectors)} vectors in batches of {batch_size}...")
        
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        print(f"\n✓ Successfully indexed {len(vectors)} vectors")
        
        # Wait a moment for index to update
        time.sleep(2)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"\nIndex Statistics:")
        print(f"  Total vectors: {stats['total_vector_count']}")
        if namespace:
            print(f"  Namespace: {namespace}")
        print(f"  Dimension: {stats['dimension']}")
    
    def search(
        self, 
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True
    ):
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filter (e.g., {"category": "travel"})
            namespace: Namespace to search in
            include_metadata: Whether to include metadata in results
        
        Returns:
            Search results
        """
        index = self.get_index()
        
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            namespace=namespace,
            include_metadata=include_metadata
        )
        
        return results
    
    def delete_all(self, namespace: str = ""):
        """
        Delete all vectors from the index.
        
        Args:
            namespace: Namespace to delete from (empty string for default)
        """
        index = self.get_index()
        index.delete(delete_all=True, namespace=namespace)
        print(f"✓ Deleted all vectors from index '{self.index_name}'")
    
    def delete_index(self):
        """
        Delete the entire index.
        """
        self.pc.delete_index(self.index_name)
        print(f"✓ Deleted index '{self.index_name}'")
    
    def get_stats(self):
        """
        Get index statistics.
        
        Returns:
            Index statistics dictionary
        """
        index = self.get_index()
        stats = index.describe_index_stats()
        return stats


def main():
    """
    Main function to demonstrate Pinecone indexing.
    """
    print("="*80)
    print("CREDIT CARD RAG - PINECONE INDEXING")
    print("="*80)
    
    # Step 1: Initialize indexer
    print("\nStep 1: Initializing Pinecone indexer...")
    indexer = PineconeIndexer(
        index_name="credit-cards",
        dimension=1024,
        metric="cosine"
    )
    
    # Step 2: Create index (if it doesn't exist)
    print("\nStep 2: Creating/connecting to index...")
    indexer.create_index(cloud="aws", region="us-east-1")
    
    # Step 3: Index embeddings
    print("\nStep 3: Indexing embeddings...")
    indexer.index_embeddings(
        embeddings_path="./embeddings/credit_cards_embeddings.json",
        batch_size=100
    )
    
    # Step 4: Display stats
    print("\nStep 4: Getting index statistics...")
    stats = indexer.get_stats()
    
    print(f"\n{'='*80}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*80}")
    print(f"Index name: {indexer.index_name}")
    print(f"Total vectors: {stats['total_vector_count']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"\n✓ All credit card embeddings are now indexed in Pinecone!")
    print(f"\nYou can now perform similarity searches and build your RAG pipeline.")
    
    # Example search (if you want to test)
    print(f"\n{'='*80}")
    print(f"EXAMPLE: Testing Search")
    print(f"{'='*80}")
    
    # Load one embedding as a test query
    with open("./embeddings/credit_cards_embeddings.json", 'r') as f:
        embedded_docs = json.load(f)
    
    if embedded_docs:
        test_vector = embedded_docs[0]["embedding"]
        print(f"\nSearching for similar cards to: {embedded_docs[0]['metadata']['card_name']}")
        
        results = indexer.search(
            query_vector=test_vector,
            top_k=3
        )
        
        print(f"\nTop 3 similar cards:")
        for i, match in enumerate(results['matches'], 1):
            print(f"\n{i}. {match['metadata']['card_name']}")
            print(f"   Bank: {match['metadata']['bank']}")
            print(f"   Category: {match['metadata']['category']}")
            print(f"   Similarity Score: {match['score']:.4f}")


if __name__ == "__main__":
    main()
