# Pinecone Indexing Guide

## Overview

This guide explains how to index your credit card embeddings into Pinecone vector database for efficient similarity search and retrieval.

## Prerequisites

1. **Embeddings Generated**: You should have already generated embeddings using `rag/embedding.py`
2. **Pinecone Account**: Sign up at [https://app.pinecone.io/](https://app.pinecone.io/)
3. **API Key**: Get your Pinecone API key from the dashboard

## Setup

### 1. Install Dependencies

```bash
uv add -r requirements.txt
```

This will install:
- `pinecone-client` - Pinecone Python SDK
- `tqdm` - Progress bar for batch uploads

### 2. Configure API Key

Add your Pinecone API key to `.env`:

```bash
# In your .env file
PINECONE_API_KEY=your_actual_pinecone_api_key_here
```

Get your API key from: [Pinecone Console](https://app.pinecone.io/) → API Keys

## Pinecone Index Configuration

### Index Specifications

- **Index Name**: `credit-cards`
- **Dimension**: 1024 (matches Mistral embed model)
- **Metric**: `cosine` (best for semantic similarity)
- **Cloud**: AWS (default)
- **Region**: `us-east-1` (default)

### Serverless vs Pod-based

The implementation uses **Serverless** indexes:
- ✅ Pay only for what you use
- ✅ Auto-scaling
- ✅ No infrastructure management
- ✅ Free tier available

## Usage

### Option 1: Run the Main Script

```bash
cd /Users/bibek/Developer/RAG/credit-card-rag
uv run rag/indexing.py
```

This will:
1. Initialize Pinecone client
2. Create index (if it doesn't exist)
3. Load embeddings from `./embeddings/credit_cards_embeddings.json`
4. Upload vectors in batches
5. Display index statistics
6. Run a test search

### Option 2: Use as a Module

```python
from rag.indexing import PineconeIndexer

# Initialize indexer
indexer = PineconeIndexer(
    index_name="credit-cards",
    dimension=1024,
    metric="cosine"
)

# Create index
indexer.create_index(cloud="aws", region="us-east-1")

# Index embeddings
indexer.index_embeddings(
    embeddings_path="./embeddings/credit_cards_embeddings.json",
    batch_size=100
)

# Get stats
stats = indexer.get_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

### Option 3: Search Vectors

```python
from rag.indexing import PineconeIndexer

indexer = PineconeIndexer()

# Search with a query vector
results = indexer.search(
    query_vector=your_query_embedding,
    top_k=5
)

# Search with metadata filter
results = indexer.search(
    query_vector=your_query_embedding,
    top_k=5,
    filter_dict={"category": "travel"}
)
```

## Metadata Structure

Each vector is stored with the following metadata:

```python
{
    "card_name": "Axis Atlas Credit Card",
    "bank": "Axis Bank",
    "category": "travel",
    "reward_type": "miles",
    "use_case": "travel",
    "source": "data/travel.md",
    "text": "1. Axis Atlas Credit Card..."  # First 1000 chars
}
```

## Metadata Filtering

You can filter searches by metadata:

```python
# Get only travel cards
filter_dict = {"category": "travel"}

# Get only Axis Bank cards
filter_dict = {"bank": "Axis Bank"}

# Get travel cards with miles rewards
filter_dict = {
    "category": "travel",
    "reward_type": "miles"
}

results = indexer.search(
    query_vector=query_vector,
    top_k=5,
    filter_dict=filter_dict
)
```

## Expected Output

```
================================================================================
CREDIT CARD RAG - PINECONE INDEXING
================================================================================

Step 1: Initializing Pinecone indexer...
✓ Initialized Pinecone client
  Index name: credit-cards
  Dimension: 1024
  Metric: cosine

Step 2: Creating/connecting to index...
✓ Index 'credit-cards' already exists

Step 3: Indexing embeddings...
================================================================================
INDEXING EMBEDDINGS TO PINECONE
================================================================================

Loading embeddings from: ./embeddings/credit_cards_embeddings.json
✓ Loaded 26 embeddings

Uploading 26 vectors in batches of 100...
Uploading batches: 100%|████████████████████| 1/1 [00:02<00:00,  2.34s/it]

✓ Successfully indexed 26 vectors

Index Statistics:
  Total vectors: 26
  Dimension: 1024

Step 4: Getting index statistics...

================================================================================
INDEXING COMPLETE
================================================================================
Index name: credit-cards
Total vectors: 26
Dimension: 1024

✓ All credit card embeddings are now indexed in Pinecone!

You can now perform similarity searches and build your RAG pipeline.
```

## Index Management

### View Index Stats

```python
indexer = PineconeIndexer()
stats = indexer.get_stats()
print(stats)
```

### Delete All Vectors

```python
indexer = PineconeIndexer()
indexer.delete_all()  # Deletes all vectors but keeps the index
```

### Delete Entire Index

```python
indexer = PineconeIndexer()
indexer.delete_index()  # Deletes the entire index
```

## Features

### ✅ Batch Processing
- Uploads vectors in configurable batches (default: 100)
- Progress bar shows upload status
- Efficient for large datasets

### ✅ Automatic Index Creation
- Creates index if it doesn't exist
- Waits for index to be ready
- Handles serverless configuration

### ✅ Metadata Support
- Stores rich metadata with each vector
- Enables filtered searches
- Preserves all chunking metadata

### ✅ Search Functionality
- Similarity search with top-k results
- Metadata filtering
- Cosine similarity scoring

### ✅ Error Handling
- Validates API key
- Handles connection errors
- Provides clear error messages

## Pinecone Free Tier

Pinecone offers a generous free tier:
- ✅ 1 serverless index
- ✅ Up to 100,000 vectors
- ✅ 2GB storage
- ✅ No credit card required

Perfect for this credit card RAG project (26 vectors)!

## Cost Estimation

For this project (26 vectors, 1024 dimensions):
- **Storage**: ~0.1 MB (well within free tier)
- **Queries**: Free tier includes generous query limits
- **Total Cost**: $0 (free tier)

## Troubleshooting

### Error: "Pinecone API key not found"
- Ensure `.env` file exists in project root
- Verify `PINECONE_API_KEY` is set correctly
- Check for typos in the key

### Error: "Index already exists"
- This is normal if you've run the script before
- The script will connect to the existing index
- Use `indexer.delete_index()` if you want to start fresh

### Error: "Dimension mismatch"
- Ensure embeddings are 1024 dimensions (Mistral embed)
- Check that index was created with dimension=1024
- Delete and recreate index if needed

### Slow Upload
- Reduce batch size (e.g., batch_size=50)
- Check internet connection
- Pinecone serverless may have rate limits

## Next Steps

After indexing, you can:

1. **Build Retriever**: Create a retriever for similarity search
2. **Implement RAG**: Connect retriever to LLM for Q&A
3. **Add Reranking**: Improve results with reranking
4. **Build API**: Create FastAPI endpoints for queries

## Integration Example

Here's how to use the indexed vectors in a RAG pipeline:

```python
from rag.indexing import PineconeIndexer
from rag.embedding import CreditCardEmbeddings

# Initialize
embedder = CreditCardEmbeddings()
indexer = PineconeIndexer()

# User query
query = "Which credit card offers the best travel rewards?"

# Generate query embedding
query_vector = embedder.embed_query(query)

# Search Pinecone
results = indexer.search(
    query_vector=query_vector,
    top_k=3,
    filter_dict={"category": "travel"}
)

# Display results
for match in results['matches']:
    print(f"Card: {match['metadata']['card_name']}")
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text'][:200]}...")
    print()
```

## Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python SDK](https://github.com/pinecone-io/pinecone-python-client)
- [Serverless Indexes Guide](https://docs.pinecone.io/guides/indexes/understanding-indexes)
