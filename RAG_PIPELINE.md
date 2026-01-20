# Credit Card RAG - Complete Pipeline Reference

## Overview

This document provides a quick reference for the complete RAG (Retrieval-Augmented Generation) pipeline for credit card recommendations.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CREDIT CARD RAG PIPELINE                      │
└─────────────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ├── Markdown Files (data/*.md)
   └── Credit Card Information

2. LOADING & CHUNKING (rag/loader_and_chunking.py)
   ├── Load markdown files
   ├── Split by credit card (1 card = 1 chunk)
   └── Extract metadata (category, bank, reward_type, etc.)

3. EMBEDDING GENERATION (rag/embedding.py)
   ├── Use Mistral mistral-embed model
   ├── Generate 1024-dim vectors
   └── Save to JSON with metadata

4. INDEXING (rag/indexing.py)
   ├── Connect to Pinecone
   ├── Create/use index
   └── Upload vectors with metadata

5. RETRIEVAL (rag/retriever.py)
   ├── Convert query to embedding
   ├── Search Pinecone (similarity + filters)
   └── Return relevant cards

6. GENERATION (Your LLM)
   ├── Format context from retrieved cards
   ├── Send to LLM with query
   └── Generate answer
```

## Quick Start Commands

### 1. Setup Environment

```bash
# Install dependencies
uv add -r requirements.txt

# Create .env file
cp .env.example .env

# Add API keys to .env
# MISTRAL_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here
```

### 2. Run Pipeline Steps

```bash
# Step 1: Load and chunk documents
uv run rag/loader_and_chunking.py

# Step 2: Generate embeddings
uv run rag/embedding.py

# Step 3: Index to Pinecone
uv run rag/indexing.py

# Step 4: Test retriever
uv run rag/retriever.py
```

## Module Reference

### loader_and_chunking.py

**Purpose**: Load markdown files and chunk by credit card

**Key Functions**:
- `extract_card_metadata(card_text, category)` - Extract metadata
- `chunk_by_credit_card(documents)` - Split into card chunks

**Output**: List of Document objects with metadata

### embedding.py

**Purpose**: Generate embeddings using Mistral

**Key Class**: `CreditCardEmbeddings`

**Methods**:
- `embed_documents(documents)` - Batch embed documents
- `embed_query(query)` - Embed single query
- `save_embeddings(embedded_docs, path)` - Save to JSON
- `load_embeddings(path)` - Load from JSON

**Output**: JSON file with embeddings and metadata

### indexing.py

**Purpose**: Index embeddings to Pinecone

**Key Class**: `PineconeIndexer`

**Methods**:
- `create_index(cloud, region)` - Create Pinecone index
- `index_embeddings(path, batch_size)` - Upload vectors
- `search(query_vector, top_k, filter_dict)` - Search index
- `get_stats()` - Get index statistics

**Output**: Vectors stored in Pinecone

### retriever.py

**Purpose**: Retrieve relevant cards for queries

**Key Class**: `CreditCardRetriever`

**Methods**:
- `retrieve(query, top_k, filter_dict)` - Basic retrieval
- `retrieve_by_category(query, category, top_k)` - Filter by category
- `retrieve_by_bank(query, bank, top_k)` - Filter by bank
- `retrieve_by_reward_type(query, reward_type, top_k)` - Filter by reward
- `retrieve_with_multiple_filters(...)` - Multiple filters
- `format_results(results)` - Format for display
- `get_context_for_llm(results)` - Format for LLM

**Output**: List of relevant documents with scores

## Code Examples

### Complete RAG Pipeline

```python
from rag.retriever import CreditCardRetriever
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
import os

# Initialize components
retriever = CreditCardRetriever()
llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# User question
question = "What's the best credit card for international travel?"

# Step 1: Retrieve relevant cards
results = retriever.retrieve(question, top_k=3)

# Step 2: Format context
context = retriever.get_context_for_llm(results, max_context_length=2000)

# Step 3: Create prompt
prompt = ChatPromptTemplate.from_template("""
You are a credit card expert. Based on the following information, 
provide a helpful recommendation.

Credit Card Information:
{context}

User Question: {question}

Provide a detailed answer with:
1. Recommended card(s)
2. Key benefits
3. Why it's suitable for the user's needs

Answer:
""")

# Step 4: Generate answer
chain = prompt | llm
response = chain.invoke({"context": context, "question": question})

print(response.content)
```

### With Filtering

```python
# Travel cards only
results = retriever.retrieve_by_category(
    query="Best card for frequent flyers",
    category="travel",
    top_k=3
)

# Axis Bank travel cards with miles
results = retriever.retrieve_with_multiple_filters(
    query="International travel benefits",
    category="travel",
    bank="Axis Bank",
    reward_type="miles",
    top_k=2
)
```

### Streaming Response

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", streaming=True)

# Retrieve context
results = retriever.retrieve(question, top_k=3)
context = retriever.get_context_for_llm(results)

# Stream response
for chunk in llm.stream(f"Context: {context}\n\nQuestion: {question}"):
    print(chunk.content, end="", flush=True)
```

## Environment Variables

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional
MISTRAL_EMBEDDING_MODEL=mistral-embed  # Default model
```

## File Structure

```
credit-card-rag/
├── data/
│   ├── travel.md           # Travel credit cards
│   ├── cashback.md         # Cashback credit cards
│   ├── reward.md           # Reward credit cards
│   └── others.md           # Other credit cards
├── rag/
│   ├── loader_and_chunking.py  # Load & chunk
│   ├── embedding.py            # Generate embeddings
│   ├── indexing.py             # Index to Pinecone
│   └── retriever.py            # Retrieve relevant cards
├── embeddings/
│   └── credit_cards_embeddings.json  # Saved embeddings
├── .env                    # API keys (gitignored)
├── .env.example            # Template
├── requirements.txt        # Dependencies
├── chunking_strategy.md    # Chunking documentation
├── EMBEDDING_GUIDE.md      # Embedding guide
├── INDEXING_GUIDE.md       # Indexing guide
└── RETRIEVER_GUIDE.md      # Retriever guide
```

## Metadata Schema

Each credit card chunk has:

```python
{
    "category": "travel",           # travel, cashback, reward, others
    "card_name": "Axis Atlas Credit Card",
    "bank": "Axis Bank",           # Axis Bank, HDFC Bank, HSBC, SBI, etc.
    "reward_type": "miles",        # miles, cashback, points, rewards
    "use_case": "travel",          # Same as category
    "source": "data/travel.md"     # Source file
}
```

## Query Examples by Use Case

### Travel Queries
```python
queries = [
    "Best credit card for international travel",
    "Card with airport lounge access",
    "Travel rewards and miles",
    "Best card for booking flights and hotels"
]
```

### Cashback Queries
```python
queries = [
    "Highest cashback on online shopping",
    "Best card for groceries and dining",
    "Cashback on utility bills",
    "Card for Swiggy and Zomato orders"
]
```

### Reward Points Queries
```python
queries = [
    "Premium card with high reward points",
    "Best card for luxury shopping",
    "Milestone benefits and bonuses",
    "High-value rewards program"
]
```

### Comparison Queries
```python
queries = [
    "Compare Axis Atlas vs HSBC TravelOne",
    "Which is better for travel: miles or cashback?",
    "Best premium card under Rs. 5000 annual fee"
]
```

## Performance Metrics

### Chunking
- **Input**: 3-4 markdown files
- **Output**: 26 credit card chunks
- **Time**: < 1 second
- **Chunk size**: 234-501 tokens

### Embedding
- **Model**: Mistral mistral-embed
- **Dimension**: 1024
- **Time**: 2-5 seconds for 26 cards
- **Cost**: < $0.01

### Indexing
- **Database**: Pinecone Serverless
- **Upload time**: 2-3 seconds
- **Storage**: ~0.1 MB
- **Cost**: Free tier

### Retrieval
- **Query time**: < 500ms
- **Top-k**: 3-5 results typical
- **Accuracy**: High (cosine similarity)

## Troubleshooting

### Common Issues

1. **"Module not found"**
   ```bash
   uv add -r requirements.txt
   ```

2. **"API key not found"**
   - Check `.env` file exists
   - Verify API keys are set
   - No quotes around keys

3. **"No results found"**
   - Ensure indexing completed
   - Check Pinecone index has data
   - Try broader query

4. **"Low similarity scores"**
   - Rephrase query
   - Remove filters
   - Check if relevant cards exist

## Next Steps

### 1. Build API (FastAPI)

```python
from fastapi import FastAPI
from rag.retriever import CreditCardRetriever

app = FastAPI()
retriever = CreditCardRetriever()

@app.get("/search")
def search(query: str, top_k: int = 5):
    results = retriever.retrieve(query, top_k)
    return {"results": results}

@app.get("/recommend")
def recommend(query: str, category: str = None):
    if category:
        results = retriever.retrieve_by_category(query, category, top_k=3)
    else:
        results = retriever.retrieve(query, top_k=3)
    
    # Format for user
    return {
        "query": query,
        "recommendations": [
            {
                "card": r['metadata']['card_name'],
                "bank": r['metadata']['bank'],
                "score": r['score']
            }
            for r in results
        ]
    }
```

### 2. Add Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query: str, top_k: int = 5):
    return retriever.retrieve(query, top_k)
```

### 3. Build Frontend

- React/Next.js for web UI
- Streamlit for quick prototype
- Gradio for demo

### 4. Add Analytics

- Track popular queries
- Monitor retrieval quality
- A/B test different strategies

## Resources

- **Chunking**: See `chunking_strategy.md`
- **Embedding**: See `EMBEDDING_GUIDE.md`
- **Indexing**: See `INDEXING_GUIDE.md`
- **Retrieval**: See `RETRIEVER_GUIDE.md`

## Support

For issues or questions:
1. Check the relevant guide
2. Review error messages
3. Verify API keys and dependencies
4. Test each component individually
