# Embedding Generation Guide

## Overview

This guide explains how to generate embeddings for credit card documents using Mistral AI's embedding model.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Mistral API key:

```
MISTRAL_API_KEY=your_actual_api_key_here
```

Get your API key from: https://console.mistral.ai/

## Mistral Embedding Model

**Model**: `mistral-embed`
- **Dimensions**: 1024
- **Max tokens**: 8192
- **Use case**: Optimized for retrieval tasks
- **Pricing**: Check [Mistral pricing](https://mistral.ai/technology/#pricing)

## Usage

### Option 1: Run the Main Script

```bash
cd /Users/bibek/Developer/RAG/credit-card-rag
python3 rag/embedding.py
```

This will:
1. Load and chunk credit card documents
2. Generate embeddings for all chunks
3. Save embeddings to `./embeddings/credit_cards_embeddings.json`

### Option 2: Use as a Module

```python
from rag.embedding import CreditCardEmbeddings
from rag.loader_and_chunking import load_and_chunk_documents

# Load and chunk documents
chunked_docs = load_and_chunk_documents()

# Initialize embedder
embedder = CreditCardEmbeddings()

# Generate embeddings
embedded_docs = embedder.embed_documents(chunked_docs)

# Save embeddings
embedder.save_embeddings(embedded_docs)
```

### Option 3: Embed a Query

```python
from rag.embedding import CreditCardEmbeddings

embedder = CreditCardEmbeddings()

# Embed a search query
query = "Which credit card offers the best travel rewards?"
query_vector = embedder.embed_query(query)

print(f"Query vector dimension: {len(query_vector)}")
```

## Output Format

Embeddings are saved as JSON with the following structure:

```json
[
  {
    "id": "card_1",
    "text": "1. Axis Atlas Credit Card\n...",
    "embedding": [0.123, -0.456, 0.789, ...],
    "metadata": {
      "category": "travel",
      "card_name": "Axis Atlas Credit Card",
      "bank": "Axis Bank",
      "reward_type": "miles",
      "use_case": "travel",
      "source": "data/travel.md"
    }
  },
  ...
]
```

## Features

### ✅ Batch Processing
- Processes all documents in a single API call
- Efficient and cost-effective

### ✅ Progress Tracking
- Shows progress during embedding generation
- Displays summary statistics

### ✅ Error Handling
- Validates API key before processing
- Provides clear error messages
- Handles API failures gracefully

### ✅ Metadata Preservation
- Maintains all chunk metadata
- Enables filtered retrieval later

### ✅ Save/Load Functionality
- Save embeddings to avoid re-generation
- Load pre-computed embeddings quickly

## Expected Output

```
================================================================================
CREDIT CARD RAG - EMBEDDING GENERATION
================================================================================

Step 1: Loading and chunking documents...
Original documents loaded: 3
Total chunks created: 26

Step 2: Initializing Mistral embeddings...
✓ Initialized Mistral Embeddings with model: mistral-embed
  Embedding dimension: 1024

Step 3: Generating embeddings...
================================================================================
GENERATING EMBEDDINGS
================================================================================
Total documents to embed: 26
Calling Mistral API...
✓ Successfully generated 26 embeddings
  Vector dimension: 1024
  Processed: 5/26 documents
  Processed: 10/26 documents
  Processed: 15/26 documents
  Processed: 20/26 documents
  Processed: 25/26 documents
  Processed: 26/26 documents

Step 4: Saving embeddings...
✓ Embeddings saved to: ./embeddings/credit_cards_embeddings.json
  File size: 1234.56 KB

================================================================================
EMBEDDING SUMMARY
================================================================================
Total documents embedded: 26
Embedding model: mistral-embed
Embedding dimension: 1024

Sample Embedding:
  ID: card_1
  Card: Axis Atlas Credit Card
  Category: travel
  Bank: Axis Bank
  Vector length: 1024
  First 5 dimensions: [0.123, -0.456, 0.789, ...]

✓ Embedding generation complete!
```

## Next Steps

After generating embeddings, you can:

1. **Store in Vector Database**: Use Pinecone, Chroma, FAISS, or Qdrant
2. **Build Retriever**: Create a retriever for similarity search
3. **Implement RAG**: Connect to LLM for question answering

## Troubleshooting

### Error: "Mistral API key not found"
- Ensure `.env` file exists in project root
- Verify `MISTRAL_API_KEY` is set correctly
- Check for typos in the key

### Error: "Module not found"
- Run `pip install -r requirements.txt`
- Ensure you're in the correct virtual environment

### Error: API rate limit
- Mistral has rate limits on free tier
- Wait a few minutes and retry
- Consider upgrading your plan

## Cost Estimation

Mistral's `mistral-embed` pricing (as of 2024):
- **Free tier**: Limited requests
- **Paid tier**: Check current pricing at https://mistral.ai/technology/#pricing

For 26 credit card chunks (~8,000 tokens total):
- Cost is minimal (typically < $0.01)

## Performance

- **Embedding time**: ~2-5 seconds for 26 documents
- **File size**: ~1-2 MB for 26 embeddings (1024 dimensions each)
- **Memory usage**: Minimal (embeddings loaded on-demand)
