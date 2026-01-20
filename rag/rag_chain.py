"""
Credit Card RAG - Complete RAG Chain
=====================================

This module implements the complete RAG (Retrieval-Augmented Generation) chain
combining retrieval from Pinecone with LLM generation using Mistral.

Features:
- Query understanding and processing
- Context retrieval from Pinecone
- LLM-based answer generation
- Streaming support
- Citation tracking
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever import CreditCardRetriever

# Load environment variables
load_dotenv()


class CreditCardRAG:
    """
    Complete RAG chain for credit card recommendations.
    """
    
    def __init__(
        self,
        llm_model: str = "mistral-large-latest",
        temperature: float = 0.3,
        top_k: int = 3,
        mistral_api_key: str = None,
        pinecone_api_key: str = None
    ):
        """
        Initialize the RAG chain.
        
        Args:
            llm_model: Mistral model to use
            temperature: LLM temperature (0-1, lower is more focused)
            top_k: Number of documents to retrieve
            mistral_api_key: Mistral API key
            pinecone_api_key: Pinecone API key
        """
        self.top_k = top_k
        
        # Initialize retriever
        self.retriever = CreditCardRetriever(
            mistral_api_key=mistral_api_key,
            pinecone_api_key=pinecone_api_key
        )
        
        # Initialize LLM
        api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.llm = ChatMistralAI(
            model=llm_model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert credit card advisor with deep knowledge of Indian credit cards. 
Your role is to provide accurate, helpful recommendations based on the user's needs.

Based on the following credit card information, answer the user's question comprehensively.

Credit Card Information:
{context}

User Question: {question}

Instructions:
1. Recommend the most suitable credit card(s) based on the context
2. Explain key benefits and features
3. Mention any fees or requirements
4. Be specific and cite card names
5. If comparing cards, highlight differences
6. Keep your answer concise but informative

Answer:
""")
        
        # Create the chain
        self.chain = (
            {
                "context": lambda x: self._get_context(x["question"], x.get("filters")),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"✓ Initialized CreditCardRAG")
        print(f"  LLM Model: {llm_model}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k retrieval: {top_k}")
    
    def _get_context(self, question: str, filters: Optional[Dict[str, str]] = None) -> str:
        """
        Retrieve context for the question.
        
        Args:
            question: User question
            filters: Optional metadata filters
        
        Returns:
            Formatted context string
        """
        if filters:
            results = self.retriever.retrieve_with_multiple_filters(
                query=question,
                category=filters.get("category"),
                bank=filters.get("bank"),
                reward_type=filters.get("reward_type"),
                top_k=self.top_k
            )
        else:
            results = self.retriever.retrieve(
                query=question,
                top_k=self.top_k
            )
        
        context = self.retriever.get_context_for_llm(
            results,
            max_context_length=3000
        )
        
        return context
    
    def ask(
        self,
        question: str,
        category: Optional[str] = None,
        bank: Optional[str] = None,
        reward_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: User question
            category: Optional category filter
            bank: Optional bank filter
            reward_type: Optional reward type filter
        
        Returns:
            Dictionary with answer and sources
        """
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if bank:
            filters["bank"] = bank
        if reward_type:
            filters["reward_type"] = reward_type
        
        # Get retrieved documents for citations
        if filters:
            retrieved_docs = self.retriever.retrieve_with_multiple_filters(
                query=question,
                category=filters.get("category"),
                bank=filters.get("bank"),
                reward_type=filters.get("reward_type"),
                top_k=self.top_k
            )
        else:
            retrieved_docs = self.retriever.retrieve(
                query=question,
                top_k=self.top_k
            )
        
        # Generate answer
        answer = self.chain.invoke({
            "question": question,
            "filters": filters if filters else None
        })
        
        # Format sources
        sources = [
            {
                "card_name": doc["metadata"]["card_name"],
                "bank": doc["metadata"]["bank"],
                "category": doc["metadata"]["category"],
                "score": round(doc["score"], 4)
            }
            for doc in retrieved_docs
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "filters_applied": filters if filters else None
        }
    
    def stream(
        self,
        question: str,
        category: Optional[str] = None,
        bank: Optional[str] = None,
        reward_type: Optional[str] = None
    ):
        """
        Stream the answer token by token.
        
        Args:
            question: User question
            category: Optional category filter
            bank: Optional bank filter
            reward_type: Optional reward type filter
        
        Yields:
            Answer tokens
        """
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if bank:
            filters["bank"] = bank
        if reward_type:
            filters["reward_type"] = reward_type
        
        # Stream answer
        for chunk in self.chain.stream({
            "question": question,
            "filters": filters if filters else None
        }):
            yield chunk
    
    def compare_cards(
        self,
        card_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple credit cards.
        
        Args:
            card_names: List of card names to compare
        
        Returns:
            Comparison answer with sources
        """
        question = f"Compare these credit cards: {', '.join(card_names)}. What are the key differences and which one is better for different use cases?"
        
        return self.ask(question)
    
    def recommend(
        self,
        use_case: str,
        budget: Optional[str] = None,
        preferences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized recommendations.
        
        Args:
            use_case: User's primary use case (e.g., "travel", "shopping")
            budget: Budget constraint (e.g., "under 5000 annual fee")
            preferences: List of preferences (e.g., ["lounge access", "no forex markup"])
        
        Returns:
            Recommendation with sources
        """
        question = f"I need a credit card for {use_case}."
        
        if budget:
            question += f" My budget is {budget}."
        
        if preferences:
            question += f" I prefer cards with: {', '.join(preferences)}."
        
        question += " What do you recommend?"
        
        return self.ask(question)


def main():
    """
    Main function to demonstrate the RAG chain.
    """
    print("="*80)
    print("CREDIT CARD RAG - COMPLETE CHAIN DEMO")
    print("="*80)
    
    # Initialize RAG chain
    print("\nInitializing RAG chain...")
    rag = CreditCardRAG(
        llm_model="mistral-large-latest",
        temperature=0.3,
        top_k=3
    )
    
    # Example 1: Simple question
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Question")
    print("="*80)
    
    result = rag.ask("What's the best credit card for international travel?")
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['card_name']} ({source['bank']}) - Score: {source['score']}")
    
    # Example 2: Filtered question
    print("\n" + "="*80)
    print("EXAMPLE 2: Filtered Question (Travel Category)")
    print("="*80)
    
    result = rag.ask(
        question="Which card offers the best lounge access?",
        category="travel"
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Filters: {result['filters_applied']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['card_name']} - Score: {source['score']}")
    
    # Example 3: Comparison
    print("\n" + "="*80)
    print("EXAMPLE 3: Card Comparison")
    print("="*80)
    
    result = rag.compare_cards([
        "Axis Atlas Credit Card",
        "HSBC TravelOne Credit Card"
    ])
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    # Example 4: Personalized recommendation
    print("\n" + "="*80)
    print("EXAMPLE 4: Personalized Recommendation")
    print("="*80)
    
    result = rag.recommend(
        use_case="online shopping and food delivery",
        budget="under 1000 annual fee",
        preferences=["high cashback", "no joining fee"]
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['card_name']} ({source['category']}) - Score: {source['score']}")
    
    # Example 5: Streaming
    print("\n" + "="*80)
    print("EXAMPLE 5: Streaming Response")
    print("="*80)
    
    question = "What are the benefits of premium credit cards?"
    print(f"\nQuestion: {question}")
    print(f"\nStreaming Answer:")
    
    for chunk in rag.stream(question):
        print(chunk, end="", flush=True)
    
    print("\n\n" + "="*80)
    print("✓ RAG Chain demo complete!")
    print("="*80)


if __name__ == "__main__":
    main()
