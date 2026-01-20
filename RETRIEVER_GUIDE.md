# Retriever Guide

## Overview

The retriever module handles the retrieval of relevant credit card information from Pinecone based on user queries. It's a crucial component of the RAG (Retrieval-Augmented Generation) pipeline.

## Architecture

```
User Query → Query Embedding → Pinecone Search → Filtered Results → Formatted Context → LLM
```

## Features

### ✅ Core Capabilities

1. **Query Embedding**: Automatically converts text queries to embeddings using Mistral
2. **Similarity Search**: Finds most relevant credit cards using cosine similarity
3. **Metadata Filtering**: Filter by category, bank, reward type, or combinations
4. **Result Formatting**: Multiple output formats for different use cases
5. **LLM Context Generation**: Prepares context optimized for language models

### ✅ Retrieval Strategies

- **Basic Retrieval**: Simple similarity search
- **Category-based**: Filter by travel/cashback/reward
- **Bank-based**: Filter by specific banks
- **Reward Type-based**: Filter by miles/cashback/points
- **Multi-filter**: Combine multiple filters

## Usage

### Basic Retrieval

```python
from rag.retriever import CreditCardRetriever

# Initialize retriever
retriever = CreditCardRetriever()

# Simple query
results = retriever.retrieve(
    query="Best credit card for travel",
    top_k=5
)

# Display results
formatted = retriever.format_results(results)
print(formatted)
```

### Filtered Retrieval

#### By Category

```python
# Get travel cards only
results = retriever.retrieve_by_category(
    query="International travel benefits",
    category="travel",
    top_k=3
)
```

#### By Bank

```python
# Get Axis Bank cards only
results = retriever.retrieve_by_bank(
    query="Rewards credit card",
    bank="Axis Bank",
    top_k=3
)
```

#### By Reward Type

```python
# Get cashback cards only
results = retriever.retrieve_by_reward_type(
    query="Shopping benefits",
    reward_type="cashback",
    top_k=3
)
```

#### Multiple Filters

```python
# Get travel cards from Axis Bank with miles rewards
results = retriever.retrieve_with_multiple_filters(
    query="Best card for frequent flyers",
    category="travel",
    bank="Axis Bank",
    reward_type="miles",
    top_k=3
)
```

### Context for LLM

```python
# Get context formatted for LLM
results = retriever.retrieve(
    query="Which credit card should I get for travel?",
    top_k=3
)

context = retriever.get_context_for_llm(
    results,
    max_context_length=2000
)

# Use context with LLM
prompt = f"""
Based on the following credit card information:

{context}

Answer this question: Which credit card should I get for travel?
"""
```

## Result Structure

Each result contains:

```python
{
    'id': 'card_1',
    'score': 0.8542,  # Similarity score (0-1, higher is better)
    'metadata': {
        'card_name': 'Axis Atlas Credit Card',
        'bank': 'Axis Bank',
        'category': 'travel',
        'reward_type': 'miles',
        'use_case': 'travel',
        'source': 'data/travel.md',
        'text': 'Full card details...'
    }
}
```

## Similarity Scores

- **0.9 - 1.0**: Highly relevant (exact match)
- **0.7 - 0.9**: Very relevant (strong match)
- **0.5 - 0.7**: Moderately relevant (good match)
- **0.3 - 0.5**: Somewhat relevant (weak match)
- **< 0.3**: Not very relevant

## Example Queries

### Travel Queries

```python
queries = [
    "Best credit card for international travel",
    "Credit card with airport lounge access",
    "Travel rewards credit card",
    "Card for frequent flyers",
    "Best card for booking flights"
]

for query in queries:
    results = retriever.retrieve(query, top_k=3)
    print(retriever.format_results(results))
```

### Cashback Queries

```python
queries = [
    "Best cashback credit card for online shopping",
    "Card with highest cashback on groceries",
    "Cashback card for dining",
    "Credit card for Swiggy and Zomato",
    "Best card for utility bill payments"
]

for query in queries:
    results = retriever.retrieve_by_category(
        query,
        category="cashback",
        top_k=3
    )
```

### Reward Points Queries

```python
queries = [
    "Premium credit card with high reward points",
    "Best card for luxury shopping",
    "High-value rewards credit card",
    "Card with milestone benefits"
]

for query in queries:
    results = retriever.retrieve_by_category(
        query,
        category="reward",
        top_k=3
    )
```

## Running the Demo

```bash
cd /Users/bibek/Developer/RAG/credit-card-rag
uv run rag/retriever.py
```

This will run several example queries and show:
1. Basic retrieval results
2. Filtered retrieval (by category, bank, reward type)
3. Multi-filter retrieval
4. LLM context generation

## Expected Output

```
================================================================================
CREDIT CARD RAG - RETRIEVER DEMO
================================================================================

Initializing retriever...
✓ Initialized CreditCardRetriever
  Index: credit-cards
  Namespace: default

================================================================================
Query: Best credit card for international travel with lounge access
Type: General travel query
================================================================================

Found 3 relevant credit card(s):

================================================================================
Result 1: Axis Atlas Credit Card
================================================================================
Relevance Score: 0.8542
Bank: Axis Bank
Category: travel
Reward Type: miles
Use Case: travel

================================================================================
Result 2: HSBC TravelOne Credit Card
================================================================================
Relevance Score: 0.8234
Bank: HSBC
Category: travel
Reward Type: cashback
Use Case: travel

...
```

## Integration with RAG Pipeline

### Step 1: Retrieve Context

```python
from rag.retriever import CreditCardRetriever

retriever = CreditCardRetriever()

# User question
question = "What's the best credit card for travel?"

# Retrieve relevant cards
results = retriever.retrieve(question, top_k=3)

# Get context for LLM
context = retriever.get_context_for_llm(results)
```

### Step 2: Generate Answer with LLM

```python
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Create prompt
prompt = ChatPromptTemplate.from_template("""
You are a credit card expert assistant. Based on the following credit card information, 
answer the user's question accurately and helpfully.

Credit Card Information:
{context}

User Question: {question}

Answer:
""")

# Generate answer
chain = prompt | llm
response = chain.invoke({
    "context": context,
    "question": question
})

print(response.content)
```

## Advanced Features

### Custom Scoring

You can implement custom scoring logic:

```python
def custom_score(results, query):
    """Apply custom scoring logic"""
    for result in results:
        # Boost travel cards for travel queries
        if "travel" in query.lower() and result['metadata']['category'] == 'travel':
            result['score'] *= 1.2
        
        # Boost cards from specific banks
        if result['metadata']['bank'] == 'Axis Bank':
            result['score'] *= 1.1
    
    # Re-sort by new scores
    return sorted(results, key=lambda x: x['score'], reverse=True)
```

### Hybrid Search

Combine semantic search with keyword matching:

```python
def hybrid_search(retriever, query, top_k=5):
    # Semantic search
    semantic_results = retriever.retrieve(query, top_k=top_k*2)
    
    # Keyword boost
    query_keywords = query.lower().split()
    for result in semantic_results:
        text = result['metadata']['text'].lower()
        keyword_matches = sum(1 for kw in query_keywords if kw in text)
        result['score'] += keyword_matches * 0.05
    
    # Re-rank and return top_k
    semantic_results.sort(key=lambda x: x['score'], reverse=True)
    return semantic_results[:top_k]
```

### Result Diversity

Ensure diverse results from different categories:

```python
def diverse_results(retriever, query, top_k=5):
    # Get more results than needed
    results = retriever.retrieve(query, top_k=top_k*3)
    
    # Select diverse results
    diverse = []
    seen_categories = set()
    
    for result in results:
        category = result['metadata']['category']
        if category not in seen_categories or len(diverse) < top_k:
            diverse.append(result)
            seen_categories.add(category)
        
        if len(diverse) >= top_k:
            break
    
    return diverse
```

## Performance Tips

### 1. Adjust top_k

```python
# For quick answers: top_k=1-3
results = retriever.retrieve(query, top_k=1)

# For comprehensive answers: top_k=5-10
results = retriever.retrieve(query, top_k=10)
```

### 2. Use Filters

```python
# Faster and more accurate
results = retriever.retrieve_by_category(
    query="travel card",
    category="travel",
    top_k=3
)

# vs unfiltered search
results = retriever.retrieve(query="travel card", top_k=3)
```

### 3. Limit Context Length

```python
# For faster LLM processing
context = retriever.get_context_for_llm(
    results,
    max_context_length=1000  # Smaller context
)
```

## Troubleshooting

### No Results Found

```python
results = retriever.retrieve(query="some query", top_k=5)

if not results:
    print("No results found. Try:")
    print("1. Broadening your query")
    print("2. Removing filters")
    print("3. Checking if index has data")
```

### Low Similarity Scores

If all scores are < 0.5:
- Query might be too vague
- Try rephrasing the query
- Check if relevant cards exist in the index

### Filter Returns Nothing

```python
# Check available values
results = retriever.retrieve("any query", top_k=100)
categories = set(r['metadata']['category'] for r in results)
banks = set(r['metadata']['bank'] for r in results)

print(f"Available categories: {categories}")
print(f"Available banks: {banks}")
```

## Next Steps

After setting up the retriever:

1. **Build RAG Chain**: Combine retriever with LLM
2. **Create API**: FastAPI endpoints for queries
3. **Add Caching**: Cache frequent queries
4. **Implement Feedback**: Track which results are helpful
5. **Build UI**: Create a web interface

## Complete RAG Example

```python
from rag.retriever import CreditCardRetriever
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

# Initialize components
retriever = CreditCardRetriever()
llm = ChatMistralAI(model="mistral-large-latest")

# Create RAG chain
def answer_question(question: str) -> str:
    # Retrieve context
    results = retriever.retrieve(question, top_k=3)
    context = retriever.get_context_for_llm(results)
    
    # Generate answer
    prompt = ChatPromptTemplate.from_template("""
    Based on the following credit card information, answer the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return response.content

# Use it
answer = answer_question("What's the best credit card for travel?")
print(answer)
```

## Resources

- [Pinecone Query Documentation](https://docs.pinecone.io/guides/data/query-data)
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
