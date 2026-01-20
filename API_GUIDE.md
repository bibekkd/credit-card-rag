# Credit Card RAG API - Complete Guide

## Overview

The Credit Card RAG API provides AI-powered credit card recommendations using Retrieval-Augmented Generation (RAG). It combines semantic search with large language models to answer questions about credit cards.

## Quick Start

### 1. Start the API Server

```bash
cd /Users/bibek/Developer/RAG/credit-card-rag
uv run api.py
```

The API will be available at: `http://localhost:8000`

### 2. Access Documentation

Interactive API documentation (Swagger UI):
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and all components are operational.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "retriever": "operational",
    "llm": "operational",
    "pinecone": "connected"
  }
}
```

### 2. Ask a Question

**POST** `/ask`

Ask any question about credit cards and get an AI-generated answer with sources.

**Request Body:**
```json
{
  "question": "What's the best credit card for international travel?",
  "category": "travel",
  "bank": null,
  "reward_type": null,
  "stream": false
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the best credit card for international travel?",
    "category": "travel"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What's the best credit card for international travel?",
        "category": "travel"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"- {source['card_name']} ({source['bank']})")
```

**Response:**
```json
{
  "question": "What's the best credit card for international travel?",
  "answer": "Based on the available options, the Axis Atlas Credit Card is an excellent choice for international travel. Here's why:\n\n1. **High Rewards on Travel**: You earn 5 EDGE Miles per Rs. 100 spent on monthly travel spends up to Rs. 2 Lakh, which translates to a 10% value-back when redeemed for travel.\n\n2. **Lounge Access**: The card offers up to 12 international lounge visits and 18 domestic lounge visits per year, making your travel experience more comfortable.\n\n3. **Milestone Benefits**: You can earn up to 5,000 bonus EDGE Miles based on annual spends.\n\nHowever, note that the card has a joining fee of Rs. 5,000 and charges a 3.5% forex markup fee.\n\nAlternatively, the HSBC TravelOne Credit Card is also worth considering if you prefer cashback-style rewards with 4 reward points per Rs. 100 on travel spends.",
  "sources": [
    {
      "card_name": "Axis Atlas Credit Card",
      "bank": "Axis Bank",
      "category": "travel",
      "score": 0.8542
    },
    {
      "card_name": "HSBC TravelOne Credit Card",
      "bank": "HSBC",
      "category": "travel",
      "score": 0.8234
    },
    {
      "card_name": "Axis Bank Horizon Credit Card",
      "bank": "Axis Bank",
      "category": "travel",
      "score": 0.7891
    }
  ],
  "filters_applied": {
    "category": "travel"
  }
}
```

### 3. Get Recommendations

**POST** `/recommend`

Get personalized credit card recommendations based on use case, budget, and preferences.

**Request Body:**
```json
{
  "use_case": "online shopping and food delivery",
  "budget": "under 1000 annual fee",
  "preferences": ["high cashback", "no joining fee"]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "use_case": "online shopping and food delivery",
    "budget": "under 1000 annual fee",
    "preferences": ["high cashback", "no joining fee"]
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "use_case": "online shopping and food delivery",
        "budget": "under 1000 annual fee",
        "preferences": ["high cashback", "no joining fee"]
    }
)

result = response.json()
print(result['answer'])
```

### 4. Compare Cards

**POST** `/compare`

Compare multiple credit cards to understand their differences.

**Request Body:**
```json
{
  "card_names": [
    "Axis Atlas Credit Card",
    "HSBC TravelOne Credit Card"
  ]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "card_names": [
      "Axis Atlas Credit Card",
      "HSBC TravelOne Credit Card"
    ]
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/compare",
    json={
        "card_names": [
            "Axis Atlas Credit Card",
            "HSBC TravelOne Credit Card"
        ]
    }
)

result = response.json()
print(result['answer'])
```

### 5. Search Cards

**GET** `/search`

Search for credit cards without LLM generation. Returns raw search results with similarity scores.

**Query Parameters:**
- `query` (required): Search query
- `category` (optional): Filter by category
- `bank` (optional): Filter by bank
- `reward_type` (optional): Filter by reward type
- `top_k` (optional): Number of results (default: 5, max: 20)

**cURL Example:**
```bash
curl "http://localhost:8000/search?query=travel+rewards&category=travel&top_k=3"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/search",
    params={
        "query": "travel rewards",
        "category": "travel",
        "top_k": 3
    }
)

results = response.json()
for card in results['results']:
    print(f"{card['card_name']} - Score: {card['score']}")
```

**Response:**
```json
{
  "query": "travel rewards",
  "filters": {
    "category": "travel",
    "bank": null,
    "reward_type": null
  },
  "total_results": 3,
  "results": [
    {
      "id": "card_14",
      "card_name": "Axis Atlas Credit Card",
      "bank": "Axis Bank",
      "category": "travel",
      "reward_type": "miles",
      "score": 0.8542,
      "text": "1. Axis Atlas Credit Card..."
    }
  ]
}
```

### 6. Get Metadata

**GET** `/categories` - Get available categories
**GET** `/banks` - Get available banks
**GET** `/reward-types` - Get available reward types

**Examples:**
```bash
curl http://localhost:8000/categories
curl http://localhost:8000/banks
curl http://localhost:8000/reward-types
```

## Filtering Options

### By Category

```json
{
  "question": "Best card for shopping?",
  "category": "cashback"
}
```

Available categories:
- `travel`
- `cashback`
- `reward`
- `others`

### By Bank

```json
{
  "question": "Show me Axis Bank cards",
  "bank": "Axis Bank"
}
```

Available banks:
- `Axis Bank`
- `HDFC Bank`
- `HSBC`
- `SBI`
- `ICICI Bank`

### By Reward Type

```json
{
  "question": "Cards with miles rewards",
  "reward_type": "miles"
}
```

Available reward types:
- `miles`
- `cashback`
- `points`
- `rewards`

### Multiple Filters

```json
{
  "question": "Best option for me?",
  "category": "travel",
  "bank": "Axis Bank",
  "reward_type": "miles"
}
```

## Streaming Responses

Enable streaming for real-time token generation:

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What are the benefits of premium credit cards?",
        "stream": True
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        print(chunk, end='', flush=True)
```

## Example Use Cases

### 1. Travel Card Recommendation

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "I travel internationally 3-4 times a year. Which credit card should I get?",
        "category": "travel"
    }
)

print(response.json()['answer'])
```

### 2. Cashback for Online Shopping

```python
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "use_case": "online shopping on Amazon and Flipkart",
        "budget": "under 500 annual fee",
        "preferences": ["high cashback", "easy redemption"]
    }
)

print(response.json()['answer'])
```

### 3. Compare Premium Cards

```python
response = requests.post(
    "http://localhost:8000/compare",
    json={
        "card_names": [
            "HDFC Infinia Credit Card",
            "Axis Magnus Credit Card"
        ]
    }
)

print(response.json()['answer'])
```

### 4. Find Cards by Bank

```python
response = requests.get(
    "http://localhost:8000/search",
    params={
        "query": "rewards program",
        "bank": "HDFC Bank",
        "top_k": 5
    }
)

for card in response.json()['results']:
    print(f"{card['card_name']} - {card['reward_type']}")
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Example Error Handling (Python)

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": "Best card?"}
    )
    response.raise_for_status()
    result = response.json()
    print(result['answer'])
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.json()}")
    
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

## Performance

### Response Times

- **Search** (`/search`): ~200-500ms
- **Ask** (`/ask`): ~2-5 seconds (includes LLM generation)
- **Recommend** (`/recommend`): ~2-5 seconds
- **Compare** (`/compare`): ~3-6 seconds

### Rate Limiting

Currently no rate limiting is implemented. In production, consider:
- Rate limiting per IP
- API key authentication
- Request throttling

## Deployment

### Local Development

```bash
uv run api.py
```

### Production (with Gunicorn)

```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Ensure these are set:
```bash
MISTRAL_API_KEY=your_key
PINECONE_API_KEY=your_key
```

## Testing

### Manual Testing

Use the interactive docs at http://localhost:8000/docs

### Automated Testing

See `test_api.py` for automated test examples.

## CORS Configuration

The API allows all origins by default. For production, update:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Security Considerations

For production deployment:

1. **API Key Authentication**: Add API key validation
2. **Rate Limiting**: Implement request throttling
3. **Input Validation**: Validate and sanitize inputs
4. **HTTPS**: Use HTTPS in production
5. **CORS**: Restrict allowed origins
6. **Logging**: Add comprehensive logging
7. **Monitoring**: Set up health checks and alerts

## Troubleshooting

### API won't start

- Check API keys in `.env`
- Ensure Pinecone index exists
- Verify dependencies installed

### Slow responses

- Check network connection
- Verify Pinecone region
- Consider caching frequent queries

### No results returned

- Check if Pinecone index has data
- Verify filters are correct
- Try broader queries

## Next Steps

1. **Add Authentication**: Implement API key or OAuth
2. **Add Caching**: Cache frequent queries with Redis
3. **Add Analytics**: Track usage and popular queries
4. **Build Frontend**: Create a web UI
5. **Add Monitoring**: Set up logging and metrics
6. **Deploy**: Deploy to cloud (AWS, GCP, Azure)

## Resources

- **API Docs**: http://localhost:8000/docs
- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
