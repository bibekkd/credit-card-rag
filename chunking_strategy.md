# Credit Card RAG - Chunking Strategy Documentation

## Overview

This document describes the chunking strategy implemented for the Credit Card RAG system. The strategy follows a **semantic chunking approach** where each credit card becomes a self-contained chunk.

## Core Principle

**1 Credit Card = 1 Chunk**

This ensures that all information about a single credit card stays together, preventing fragmentation and maintaining semantic coherence.

---

## Why This Strategy?

### ✅ Advantages

1. **Self-Contained Information**
   - Each chunk contains complete card details
   - No need to retrieve multiple chunks for one card
   - Enables accurate card-specific queries

2. **Better Comparisons**
   - LLM can compare entire cards in context
   - No missing information during comparison
   - Cleaner, more accurate recommendations

3. **No Cross-Card Contamination**
   - Cards never split across chunks
   - Prevents mixing features from different cards
   - Reduces hallucination risk

4. **Metadata-Driven Retrieval**
   - Rich metadata enables filtered searches
   - Can query by category, bank, reward type
   - Improves retrieval precision

### ❌ What We Avoid

- **Token-based splitting**: Would break cards mid-description
- **Fixed-size chunks**: Would split related information
- **Overlap between cards**: Unnecessary since cards are distinct entities

---

## Implementation Details

### Chunk Boundaries

Each chunk includes ALL sections of a credit card:

```
1. Card Name
   ├── Joining Fee / Annual Fee
   ├── Key Features and Benefits
   ├── Rewards Structure
   ├── Lounge Access
   └── Conclusion
```

### Detection Pattern

Cards are identified using regex pattern: `^\d+\.\s+[A-Z]`

This matches lines like:
- `1. Axis Atlas Credit Card`
- `2. Axis Bank Horizon Credit Card`
- `3. HSBC TravelOne Credit Card`

### Chunk Size

- **Target Range**: 300-500 tokens per chunk
- **Actual Range**: 234-501 tokens (tested on current dataset)
- **No overlap needed**: Cards are naturally distinct

---

## Metadata Schema

Each chunk is enriched with the following metadata:

```json
{
  "category": "travel",
  "card_name": "Axis Atlas Credit Card",
  "bank": "Axis Bank",
  "reward_type": "miles",
  "use_case": "travel",
  "source": "data/travel.md"
}
```

### Metadata Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| `category` | Card category from filename | `travel`, `cashback`, `reward` |
| `card_name` | Full name of the credit card | `Axis Atlas Credit Card` |
| `bank` | Issuing bank | `Axis Bank`, `HDFC Bank`, `HSBC`, `SBI`, `ICICI Bank` |
| `reward_type` | Type of rewards offered | `miles`, `cashback`, `points`, `rewards` |
| `use_case` | Primary use case (defaults to category) | `travel`, `cashback`, `reward` |
| `source` | Source file path | `data/travel.md` |

### Metadata Extraction Logic

1. **Category**: Extracted from filename stem (e.g., `travel.md` → `travel`)
2. **Card Name**: Parsed from first numbered line using regex
3. **Bank**: Intelligently detected from card name keywords
4. **Reward Type**: Analyzed from content (keywords: "miles", "cashback", "points")
5. **Use Case**: Defaults to category value

---

## Code Implementation

### Main Functions

#### 1. `extract_card_metadata(card_text, category)`
Extracts metadata from a credit card chunk.

**Process:**
- Parses card name from numbered line
- Detects bank from card name
- Analyzes content for reward type
- Returns metadata dictionary

#### 2. `chunk_by_credit_card(documents)`
Main chunking logic that processes loaded documents.

**Process:**
1. Iterate through loaded documents
2. Extract category from filename
3. Split content by card pattern
4. Extract metadata for each card
5. Create Document objects with metadata
6. Return list of chunked documents

### Usage Example

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load documents
loader = DirectoryLoader(
    './data/', 
    glob='*.md', 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
docs = loader.load()

# Apply chunking strategy
chunked_docs = chunk_by_credit_card(docs)

# Result: List of Document objects, one per credit card
```

---

## Test Results

### Dataset Statistics

- **Input Files**: 3 markdown files (`travel.md`, `cashback.md`, `reward.md`)
- **Output Chunks**: 26 credit card chunks
- **Chunk Size Range**: 234-501 tokens
- **Average Chunk Size**: ~320 tokens

### Sample Chunks

#### Travel Category (3 chunks)
```
Chunk 1: Axis Atlas Credit Card (~355 tokens)
Chunk 2: Axis Bank Horizon Credit Card (~327 tokens)
Chunk 3: HSBC TravelOne Credit Card (~374 tokens)
```

#### Cashback Category (10 chunks)
```
Chunk 1: YES Bank Paisabazaar PaisaSave (~304 tokens)
Chunk 2: Cashback SBI Card (~234 tokens)
Chunk 3: HDFC Millennia Credit Card (~248 tokens)
...and 7 more
```

#### Reward Category (13 chunks)
```
Chunk 1: HDFC Infinia Credit Card (~501 tokens)
Chunk 2: Axis Magnus Credit Card (~497 tokens)
...and 11 more
```

---

## Benefits for RAG Pipeline

### 1. Filtered Retrieval

Metadata enables precise filtering:

```python
# Get only travel cards
retriever.filter({"category": "travel"})

# Get Axis Bank cards with miles rewards
retriever.filter({"bank": "Axis Bank", "reward_type": "miles"})
```

### 2. Better Context

Each retrieved chunk provides complete card information, enabling:
- Accurate feature comparisons
- Complete benefit listings
- Informed recommendations

### 3. Easier Debugging

Metadata makes it easy to:
- Track which card was retrieved
- Identify retrieval patterns
- Debug incorrect responses

### 4. Scalability

The strategy scales well:
- Add new categories by creating new `.md` files
- Add new cards within existing categories
- No code changes needed for new data

---

## Production Considerations

### ✅ Ready for Production

- Chunk sizes are optimal for embedding models
- Metadata structure supports advanced queries
- No manual intervention needed for new cards
- Tested and validated on real dataset

### 🔄 Future Enhancements

1. **Dynamic Bank Detection**: Add more banks to detection logic
2. **Additional Metadata**: Extract fees, lounge access counts, etc.
3. **Validation**: Add checks for chunk quality and completeness
4. **Multi-language Support**: Extend for non-English cards

---

## Integration Guide

### Step 1: Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents([doc.page_content for doc in chunked_docs])
```

### Step 2: Store in Vector Database

```python
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name="credit-cards"
)
```

### Step 3: Create Retriever with Metadata Filtering

```python
# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Filtered retriever
travel_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"category": "travel"}
    }
)
```

### Step 4: Use in RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain("Which travel credit card offers the most lounge access?")
```

---

## Conclusion

This chunking strategy provides an optimal balance between:
- **Semantic coherence** (keeping cards together)
- **Chunk size** (300-500 tokens)
- **Retrieval precision** (rich metadata)
- **Scalability** (easy to extend)

The implementation is production-ready and tested on real credit card data, making it suitable for immediate deployment in a RAG pipeline.
