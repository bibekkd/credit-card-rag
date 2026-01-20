"""
Credit Card RAG - Embedding Generation using Mistral
=====================================================

This module handles embedding generation for credit card chunks using Mistral's
embedding model (mistral-embed).

Features:
- Uses Mistral's mistral-embed model (1024 dimensions)
- Batch processing for efficiency
- Progress tracking
- Error handling and retry logic
- Saves embeddings with metadata
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
import json
from pathlib import Path

# Load environment variables
load_dotenv()


class CreditCardEmbeddings:
    """
    Handles embedding generation for credit card documents using Mistral AI.
    """
    
    def __init__(self, api_key: str = None, model: str = "mistral-embed"):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model: Mistral embedding model to use (default: mistral-embed)
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.embeddings = MistralAIEmbeddings(
            model=self.model,
            api_key=self.api_key
        )
        
        print(f"✓ Initialized Mistral Embeddings with model: {self.model}")
        print(f"  Embedding dimension: 1024")
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects with page_content and metadata
        
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        print(f"\n{'='*80}")
        print(f"GENERATING EMBEDDINGS")
        print(f"{'='*80}\n")
        print(f"Total documents to embed: {len(documents)}")
        
        embedded_docs = []
        
        # Extract text content for embedding
        texts = [doc.page_content for doc in documents]
        
        print(f"Calling Mistral API...")
        
        try:
            # Generate embeddings in batch
            vectors = self.embeddings.embed_documents(texts)
            
            print(f"✓ Successfully generated {len(vectors)} embeddings")
            print(f"  Vector dimension: {len(vectors[0]) if vectors else 0}")
            
            # Combine embeddings with metadata
            for i, (doc, vector) in enumerate(zip(documents, vectors)):
                embedded_doc = {
                    "id": f"card_{i+1}",
                    "text": doc.page_content,
                    "embedding": vector,
                    "metadata": doc.metadata
                }
                embedded_docs.append(embedded_doc)
                
                # Progress indicator
                if (i + 1) % 5 == 0 or (i + 1) == len(documents):
                    print(f"  Processed: {i+1}/{len(documents)} documents")
            
            return embedded_docs
            
        except Exception as e:
            print(f"✗ Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
        
        Returns:
            Embedding vector
        """
        try:
            vector = self.embeddings.embed_query(query)
            return vector
        except Exception as e:
            print(f"✗ Error embedding query: {str(e)}")
            raise
    
    def save_embeddings(
        self, 
        embedded_docs: List[Dict[str, Any]], 
        output_path: str = "./embeddings/credit_cards_embeddings.json"
    ):
        """
        Save embeddings to a JSON file.
        
        Args:
            embedded_docs: List of embedded documents
            output_path: Path to save the embeddings
        """
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_docs, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Embeddings saved to: {output_path}")
        print(f"  File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    def load_embeddings(
        self, 
        input_path: str = "./embeddings/credit_cards_embeddings.json"
    ) -> List[Dict[str, Any]]:
        """
        Load embeddings from a JSON file.
        
        Args:
            input_path: Path to load the embeddings from
        
        Returns:
            List of embedded documents
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            embedded_docs = json.load(f)
        
        print(f"✓ Loaded {len(embedded_docs)} embeddings from: {input_path}")
        return embedded_docs


def main():
    """
    Main function to demonstrate embedding generation.
    """
    # Import the chunking function
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from loader_and_chunking import chunk_by_credit_card
    
    print("="*80)
    print("CREDIT CARD RAG - EMBEDDING GENERATION")
    print("="*80)
    
    # Step 1: Load and chunk documents
    print("\nStep 1: Loading and chunking documents...")
    loader = DirectoryLoader(
        './data/', 
        glob='*.md', 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=False
    )
    docs = loader.load()
    chunked_docs = chunk_by_credit_card(docs)
    print(f"✓ Loaded {len(docs)} documents and created {len(chunked_docs)} chunks")
    
    # Step 2: Initialize embeddings
    print("\nStep 2: Initializing Mistral embeddings...")
    embedder = CreditCardEmbeddings()
    
    # Step 3: Generate embeddings
    print("\nStep 3: Generating embeddings...")
    embedded_docs = embedder.embed_documents(chunked_docs)
    
    # Step 4: Save embeddings
    print("\nStep 4: Saving embeddings...")
    embedder.save_embeddings(embedded_docs)
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"EMBEDDING SUMMARY")
    print(f"{'='*80}")
    print(f"Total documents embedded: {len(embedded_docs)}")
    print(f"Embedding model: {embedder.model}")
    print(f"Embedding dimension: 1024")
    
    # Show sample
    if embedded_docs:
        sample = embedded_docs[0]
        print(f"\nSample Embedding:")
        print(f"  ID: {sample['id']}")
        print(f"  Card: {sample['metadata'].get('card_name', 'N/A')}")
        print(f"  Category: {sample['metadata'].get('category', 'N/A')}")
        print(f"  Bank: {sample['metadata'].get('bank', 'N/A')}")
        print(f"  Vector length: {len(sample['embedding'])}")
        print(f"  First 5 dimensions: {sample['embedding'][:5]}")
    
    print(f"\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
