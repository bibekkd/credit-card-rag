"""
Credit Card RAG - Retriever
============================

This module handles retrieval of relevant credit card information from Pinecone
vector database based on user queries.

Features:
- Query embedding generation
- Similarity search in Pinecone
- Metadata filtering
- Result reranking
- Multiple retrieval strategies
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from rag.embedding import CreditCardEmbeddings
import json

# Load environment variables
load_dotenv()


class CreditCardRetriever:
    """
    Handles retrieval of credit card information from Pinecone.
    """
    
    def __init__(
        self,
        index_name: str = "credit-cards",
        mistral_api_key: str = None,
        pinecone_api_key: str = None,
        namespace: str = ""
    ):
        """
        Initialize the retriever.
        
        Args:
            index_name: Name of the Pinecone index
            mistral_api_key: Mistral API key for embeddings
            pinecone_api_key: Pinecone API key
            namespace: Pinecone namespace (optional)
        """
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize embedding generator
        self.embedder = CreditCardEmbeddings(api_key=mistral_api_key)
        
        # Initialize Pinecone
        pinecone_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError(
                "Pinecone API key not found. Set PINECONE_API_KEY environment variable "
                "or pass pinecone_api_key parameter."
            )
        
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)
        
        print(f"✓ Initialized CreditCardRetriever")
        print(f"  Index: {self.index_name}")
        print(f"  Namespace: {self.namespace or 'default'}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant credit cards based on a query.
        
        Args:
            query: User query string
            top_k: Number of results to return
            filter_dict: Metadata filter (e.g., {"category": "travel"})
            include_scores: Whether to include similarity scores
        
        Returns:
            List of retrieved documents with metadata and scores
        """
        # Generate query embedding
        query_vector = self.embedder.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            namespace=self.namespace,
            include_metadata=True
        )
        
        # Format results
        documents = []
        for match in results['matches']:
            doc = {
                'id': match['id'],
                'score': match['score'] if include_scores else None,
                'metadata': match['metadata']
            }
            documents.append(doc)
        
        return documents
    
    def retrieve_by_category(
        self,
        query: str,
        category: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve credit cards filtered by category.
        
        Args:
            query: User query string
            category: Category to filter by (e.g., "travel", "cashback", "reward")
            top_k: Number of results to return
        
        Returns:
            List of retrieved documents
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict={"category": category}
        )
    
    def retrieve_by_bank(
        self,
        query: str,
        bank: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve credit cards filtered by bank.
        
        Args:
            query: User query string
            bank: Bank name to filter by (e.g., "Axis Bank", "HDFC Bank")
            top_k: Number of results to return
        
        Returns:
            List of retrieved documents
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict={"bank": bank}
        )
    
    def retrieve_by_reward_type(
        self,
        query: str,
        reward_type: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve credit cards filtered by reward type.
        
        Args:
            query: User query string
            reward_type: Reward type to filter by (e.g., "miles", "cashback", "points")
            top_k: Number of results to return
        
        Returns:
            List of retrieved documents
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict={"reward_type": reward_type}
        )
    
    def retrieve_with_multiple_filters(
        self,
        query: str,
        category: Optional[str] = None,
        bank: Optional[str] = None,
        reward_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve credit cards with multiple filters.
        
        Args:
            query: User query string
            category: Category filter (optional)
            bank: Bank filter (optional)
            reward_type: Reward type filter (optional)
            top_k: Number of results to return
        
        Returns:
            List of retrieved documents
        """
        # Build filter dictionary
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if bank:
            filter_dict["bank"] = bank
        if reward_type:
            filter_dict["reward_type"] = reward_type
        
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict if filter_dict else None
        )
    
    def format_results(
        self,
        results: List[Dict[str, Any]],
        include_text: bool = True,
        max_text_length: int = 500
    ) -> str:
        """
        Format retrieval results as a readable string.
        
        Args:
            results: List of retrieved documents
            include_text: Whether to include card text
            max_text_length: Maximum length of text to include
        
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        output = []
        output.append(f"Found {len(results)} relevant credit card(s):\n")
        
        for i, doc in enumerate(results, 1):
            metadata = doc['metadata']
            score = doc.get('score')
            
            output.append(f"\n{'='*80}")
            output.append(f"Result {i}: {metadata.get('card_name', 'Unknown Card')}")
            output.append(f"{'='*80}")
            
            if score is not None:
                output.append(f"Relevance Score: {score:.4f}")
            
            output.append(f"Bank: {metadata.get('bank', 'N/A')}")
            output.append(f"Category: {metadata.get('category', 'N/A')}")
            output.append(f"Reward Type: {metadata.get('reward_type', 'N/A')}")
            output.append(f"Use Case: {metadata.get('use_case', 'N/A')}")
            
            if include_text and 'text' in metadata:
                text = metadata['text']
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "..."
                output.append(f"\nDetails:\n{text}")
        
        return "\n".join(output)
    
    def get_context_for_llm(
        self,
        results: List[Dict[str, Any]],
        max_context_length: int = 2000
    ) -> str:
        """
        Format retrieval results as context for LLM.
        
        Args:
            results: List of retrieved documents
            max_context_length: Maximum total context length
        
        Returns:
            Formatted context string for LLM
        """
        if not results:
            return "No relevant credit cards found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(results, 1):
            metadata = doc['metadata']
            
            # Create context for this card
            card_context = f"""
Credit Card {i}: {metadata.get('card_name', 'Unknown')}
Bank: {metadata.get('bank', 'N/A')}
Category: {metadata.get('category', 'N/A')}
Reward Type: {metadata.get('reward_type', 'N/A')}

Details:
{metadata.get('text', 'No details available')}
"""
            
            # Check if adding this would exceed max length
            if current_length + len(card_context) > max_context_length:
                break
            
            context_parts.append(card_context)
            current_length += len(card_context)
        
        return "\n---\n".join(context_parts)


def main():
    """
    Main function to demonstrate retriever usage.
    """
    print("="*80)
    print("CREDIT CARD RAG - RETRIEVER DEMO")
    print("="*80)
    
    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = CreditCardRetriever()
    
    # Example queries
    queries = [
        {
            "query": "Best credit card for international travel with lounge access",
            "description": "General travel query"
        },
        {
            "query": "Cashback credit card for online shopping",
            "description": "Cashback query"
        },
        {
            "query": "Premium credit card with high reward points",
            "description": "Reward points query"
        }
    ]
    
    for example in queries:
        query = example["query"]
        description = example["description"]
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Type: {description}")
        print(f"{'='*80}")
        
        # Retrieve results
        results = retriever.retrieve(query=query, top_k=3)
        
        # Format and display
        formatted = retriever.format_results(results, include_text=False)
        print(formatted)
    
    # Example with filters
    print(f"\n{'='*80}")
    print("FILTERED RETRIEVAL EXAMPLES")
    print(f"{'='*80}")
    
    # Travel cards only
    print("\n1. Travel Cards Only:")
    results = retriever.retrieve_by_category(
        query="Best card for frequent flyers",
        category="travel",
        top_k=2
    )
    print(retriever.format_results(results, include_text=False))
    
    # Axis Bank cards only
    print("\n2. Axis Bank Cards Only:")
    results = retriever.retrieve_by_bank(
        query="Rewards credit card",
        bank="Axis Bank",
        top_k=2
    )
    print(retriever.format_results(results, include_text=False))
    
    # Multiple filters
    print("\n3. Travel Cards with Miles Rewards:")
    results = retriever.retrieve_with_multiple_filters(
        query="International travel benefits",
        category="travel",
        reward_type="miles",
        top_k=2
    )
    print(retriever.format_results(results, include_text=False))
    
    # Example: Context for LLM
    print(f"\n{'='*80}")
    print("CONTEXT FOR LLM")
    print(f"{'='*80}")
    
    query = "Which credit card should I get for travel?"
    results = retriever.retrieve(query=query, top_k=2)
    context = retriever.get_context_for_llm(results, max_context_length=1000)
    
    print(f"\nQuery: {query}")
    print(f"\nGenerated Context for LLM:")
    print(context)
    
    print(f"\n{'='*80}")
    print("✓ Retriever demo complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
