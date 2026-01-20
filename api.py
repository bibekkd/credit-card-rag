"""
Credit Card RAG - FastAPI Application
======================================

This module provides REST API endpoints for the credit card RAG system.

Endpoints:
- POST /ask - Ask a question
- POST /recommend - Get personalized recommendations
- POST /compare - Compare credit cards
- GET /search - Search credit cards
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from rag.rag_chain import CreditCardRAG
import json

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card RAG API",
    description="AI-powered credit card recommendation system using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chain (singleton)
rag_chain = None


def get_rag_chain() -> CreditCardRAG:
    """Get or create RAG chain instance."""
    global rag_chain
    if rag_chain is None:
        rag_chain = CreditCardRAG(
            llm_model="mistral-large-latest",
            temperature=0.3,
            top_k=3
        )
    return rag_chain


# Pydantic models for request/response
class AskRequest(BaseModel):
    question: str = Field(..., description="User's question about credit cards")
    category: Optional[str] = Field(None, description="Filter by category (travel, cashback, reward)")
    bank: Optional[str] = Field(None, description="Filter by bank name")
    reward_type: Optional[str] = Field(None, description="Filter by reward type (miles, cashback, points)")
    stream: bool = Field(False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What's the best credit card for international travel?",
                "category": "travel",
                "stream": False
            }
        }


class RecommendRequest(BaseModel):
    use_case: str = Field(..., description="Primary use case (e.g., 'travel', 'shopping')")
    budget: Optional[str] = Field(None, description="Budget constraint (e.g., 'under 5000 annual fee')")
    preferences: Optional[List[str]] = Field(None, description="List of preferences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "use_case": "online shopping and food delivery",
                "budget": "under 1000 annual fee",
                "preferences": ["high cashback", "no joining fee"]
            }
        }


class CompareRequest(BaseModel):
    card_names: List[str] = Field(..., description="List of card names to compare", min_length=2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "card_names": [
                    "Axis Atlas Credit Card",
                    "HSBC TravelOne Credit Card"
                ]
            }
        }


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    filters_applied: Optional[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        rag = get_rag_chain()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "retriever": "operational",
                "llm": "operational",
                "pinecone": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask_question(request: AskRequest):
    """
    Ask a question about credit cards.
    
    Returns an AI-generated answer with sources.
    """
    try:
        rag = get_rag_chain()
        
        if request.stream:
            # Return streaming response
            async def generate():
                for chunk in rag.stream(
                    question=request.question,
                    category=request.category,
                    bank=request.bank,
                    reward_type=request.reward_type
                ):
                    yield chunk
            
            return StreamingResponse(generate(), media_type="text/plain")
        
        # Regular response
        result = rag.ask(
            question=request.question,
            category=request.category,
            bank=request.bank,
            reward_type=request.reward_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/recommend", response_model=AskResponse, tags=["Recommendations"])
async def get_recommendation(request: RecommendRequest):
    """
    Get personalized credit card recommendations.
    
    Provide your use case, budget, and preferences to get tailored suggestions.
    """
    try:
        rag = get_rag_chain()
        
        result = rag.recommend(
            use_case=request.use_case,
            budget=request.budget,
            preferences=request.preferences
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")


@app.post("/compare", response_model=AskResponse, tags=["Comparison"])
async def compare_cards(request: CompareRequest):
    """
    Compare multiple credit cards.
    
    Provide a list of card names to get a detailed comparison.
    """
    try:
        rag = get_rag_chain()
        
        result = rag.compare_cards(card_names=request.card_names)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing cards: {str(e)}")


@app.get("/search", tags=["Search"])
async def search_cards(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    bank: Optional[str] = Query(None, description="Filter by bank"),
    reward_type: Optional[str] = Query(None, description="Filter by reward type"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results")
):
    """
    Search for credit cards without LLM generation.
    
    Returns raw search results with similarity scores.
    """
    try:
        rag = get_rag_chain()
        
        # Get retriever results
        if category or bank or reward_type:
            results = rag.retriever.retrieve_with_multiple_filters(
                query=query,
                category=category,
                bank=bank,
                reward_type=reward_type,
                top_k=top_k
            )
        else:
            results = rag.retriever.retrieve(
                query=query,
                top_k=top_k
            )
        
        # Format results
        formatted_results = [
            {
                "id": doc["id"],
                "card_name": doc["metadata"]["card_name"],
                "bank": doc["metadata"]["bank"],
                "category": doc["metadata"]["category"],
                "reward_type": doc["metadata"]["reward_type"],
                "score": round(doc["score"], 4),
                "text": doc["metadata"]["text"][:300] + "..."  # Preview
            }
            for doc in results
        ]
        
        return {
            "query": query,
            "filters": {
                "category": category,
                "bank": bank,
                "reward_type": reward_type
            },
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching cards: {str(e)}")


@app.get("/categories", tags=["Metadata"])
async def get_categories():
    """Get available credit card categories."""
    return {
        "categories": ["travel", "cashback", "reward", "others"]
    }


@app.get("/banks", tags=["Metadata"])
async def get_banks():
    """Get available banks."""
    return {
        "banks": [
            "Axis Bank",
            "HDFC Bank",
            "HSBC",
            "SBI",
            "ICICI Bank"
        ]
    }


@app.get("/reward-types", tags=["Metadata"])
async def get_reward_types():
    """Get available reward types."""
    return {
        "reward_types": ["miles", "cashback", "points", "rewards"]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "detail": str(exc)
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG chain on startup."""
    print("="*80)
    print("CREDIT CARD RAG API - STARTING UP")
    print("="*80)
    
    try:
        # Initialize RAG chain
        get_rag_chain()
        print("\n✓ RAG chain initialized successfully")
        print("\n" + "="*80)
        print("API is ready!")
        print("Documentation: http://localhost:8000/docs")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n✗ Error initializing RAG chain: {str(e)}")
        print("Please check your API keys and Pinecone index.\n")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
