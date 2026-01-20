# Credit Card RAG System 💳

An AI-powered credit card recommendation system using Retrieval-Augmented Generation (RAG) with Mistral AI and Pinecone.

## 🌟 Features

- **Intelligent Recommendations**: Get personalized credit card suggestions based on your needs
- **Semantic Search**: Find relevant cards using natural language queries
- **Card Comparisons**: Compare multiple credit cards side-by-side
- **Metadata Filtering**: Filter by category, bank, or reward type
- **REST API**: FastAPI-based API with comprehensive endpoints
- **Streaming Support**: Real-time response streaming
- **Source Citations**: Every answer includes source cards with similarity scores

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Credit Card RAG Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

Data (Markdown) → Chunking → Embedding → Indexing → Retrieval → LLM → Answer
                     ↓           ↓          ↓          ↓         ↓
                 1 card =    Mistral    Pinecone   Semantic   Mistral
                 1 chunk     Embed      Vector DB   Search    Large
```

## 📋 Prerequisites

- Python 3.11+
- Mistral AI API key ([Get here](https://console.mistral.ai/))
- Pinecone API key ([Get here](https://app.pinecone.io/))

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd /Users/bibek/Developer/RAG/credit-card-rag

# Install dependencies
uv add -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run the Pipeline

```bash
# Step 1: Load and chunk documents
uv run rag/loader_and_chunking.py

# Step 2: Generate embeddings
uv run rag/embedding.py

# Step 3: Index to Pinecone
uv run rag/indexing.py

# Step 4: Test retriever
uv run rag/retriever.py

# Step 5: Test RAG chain
uv run rag/rag_chain.py
```

### 3. Start the API

```bash
uv run api.py
```

API will be available at: http://localhost:8000

Interactive docs: http://localhost:8000/docs

### 4. Test the API

```bash
# In a new terminal
uv run test_api.py
```

## 📁 Project Structure

```
credit-card-rag/
├── data/                          # Credit card data (markdown files)
│   ├── travel.md
│   ├── cashback.md
│   ├── reward.md
│   └── others.md
├── rag/                           # RAG pipeline modules
│   ├── loader_and_chunking.py    # Load & chunk documents
│   ├── embedding.py               # Generate embeddings
│   ├── indexing.py                # Index to Pinecone
│   ├── retriever.py               # Retrieve relevant cards
│   └── rag_chain.py               # Complete RAG chain with LLM
├── embeddings/                    # Generated embeddings
│   └── credit_cards_embeddings.json
├── api.py                         # FastAPI application
├── test_api.py                    # API test suite
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (gitignored)
├── .env.example                   # Environment template
├── chunking_strategy.md           # Chunking documentation
├── EMBEDDING_GUIDE.md             # Embedding guide
├── INDEXING_GUIDE.md              # Indexing guide
├── RETRIEVER_GUIDE.md             # Retriever guide
├── API_GUIDE.md                   # API documentation
├── RAG_PIPELINE.md                # Complete pipeline reference
└── README.md                      # This file
```

## 🔑 Environment Variables

Create a `.env` file with:

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional
MISTRAL_EMBEDDING_MODEL=mistral-embed
```

## 🎯 Usage Examples

### Python SDK

```python
from rag.rag_chain import CreditCardRAG

# Initialize RAG chain
rag = CreditCardRAG()

# Ask a question
result = rag.ask("What's the best credit card for international travel?")
print(result['answer'])

# Get recommendations
result = rag.recommend(
    use_case="online shopping",
    budget="under 1000 annual fee",
    preferences=["high cashback"]
)
print(result['answer'])

# Compare cards
result = rag.compare_cards([
    "Axis Atlas Credit Card",
    "HSBC TravelOne Credit Card"
])
print(result['answer'])
```

### REST API

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Best credit card for travel?",
    "category": "travel"
  }'

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "use_case": "online shopping",
    "budget": "under 1000 annual fee"
  }'

# Search cards
curl "http://localhost:8000/search?query=cashback&category=cashback&top_k=5"
```

### Python Requests

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "Best credit card for travel?",
        "category": "travel"
    }
)

result = response.json()
print(result['answer'])

# Print sources
for source in result['sources']:
    print(f"- {source['card_name']} ({source['bank']})")
```

## 🎨 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ask` | POST | Ask a question |
| `/recommend` | POST | Get recommendations |
| `/compare` | POST | Compare cards |
| `/search` | GET | Search cards |
| `/categories` | GET | Get categories |
| `/banks` | GET | Get banks |
| `/reward-types` | GET | Get reward types |

See [API_GUIDE.md](API_GUIDE.md) for detailed documentation.

## 🔍 Filtering Options

### By Category
- `travel` - Travel credit cards
- `cashback` - Cashback credit cards
- `reward` - Reward points cards
- `others` - Other cards

### By Bank
- Axis Bank
- HDFC Bank
- HSBC
- SBI
- ICICI Bank

### By Reward Type
- `miles` - Travel miles
- `cashback` - Cash back
- `points` - Reward points
- `rewards` - General rewards

## 📊 Technical Details

### Chunking Strategy
- **Principle**: 1 credit card = 1 chunk
- **Chunk Size**: 234-501 tokens
- **Total Chunks**: 26 credit cards
- **Metadata**: category, card_name, bank, reward_type, use_case

### Embedding Model
- **Model**: Mistral `mistral-embed`
- **Dimension**: 1024
- **Max Tokens**: 8192
- **Cost**: ~$0.01 for 26 cards

### Vector Database
- **Database**: Pinecone Serverless
- **Index**: `credit-cards`
- **Metric**: Cosine similarity
- **Region**: AWS us-east-1

### LLM
- **Model**: Mistral `mistral-large-latest`
- **Temperature**: 0.3 (focused responses)
- **Context**: Top 3 relevant cards

## 📈 Performance

- **Chunking**: < 1 second
- **Embedding**: 2-5 seconds for 26 cards
- **Indexing**: 2-3 seconds
- **Retrieval**: < 500ms
- **LLM Generation**: 2-5 seconds
- **Total Query Time**: ~3-6 seconds

## 🧪 Testing

Run the test suite:

```bash
# Start API first
uv run api.py

# In another terminal, run tests
uv run test_api.py
```

Tests include:
- Health check
- Simple questions
- Filtered queries
- Recommendations
- Comparisons
- Search functionality
- Metadata endpoints
- Error handling

## 📚 Documentation

- **[chunking_strategy.md](chunking_strategy.md)** - Chunking implementation details
- **[EMBEDDING_GUIDE.md](EMBEDDING_GUIDE.md)** - Embedding generation guide
- **[INDEXING_GUIDE.md](INDEXING_GUIDE.md)** - Pinecone indexing guide
- **[RETRIEVER_GUIDE.md](RETRIEVER_GUIDE.md)** - Retriever usage guide
- **[API_GUIDE.md](API_GUIDE.md)** - Complete API documentation
- **[RAG_PIPELINE.md](RAG_PIPELINE.md)** - Full pipeline reference

## 🛠️ Development

### Adding New Credit Cards

1. Add card details to appropriate markdown file in `data/`
2. Re-run the pipeline:
   ```bash
   uv run rag/loader_and_chunking.py
   uv run rag/embedding.py
   uv run rag/indexing.py
   ```

### Customizing the LLM

Edit `rag/rag_chain.py`:

```python
rag = CreditCardRAG(
    llm_model="mistral-large-latest",  # Change model
    temperature=0.3,                    # Adjust creativity
    top_k=3                             # Number of cards to retrieve
)
```

### Customizing Prompts

Edit the prompt template in `rag/rag_chain.py`:

```python
self.prompt = ChatPromptTemplate.from_template("""
Your custom prompt here...
{context}
{question}
""")
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t credit-card-rag .
docker run -p 8000:8000 --env-file .env credit-card-rag
```

### Production Considerations

- Add API key authentication
- Implement rate limiting
- Add request caching (Redis)
- Set up monitoring and logging
- Use HTTPS
- Configure CORS properly
- Add database for analytics

## 💰 Cost Estimation

For 26 credit cards:

| Component | Cost |
|-----------|------|
| Mistral Embeddings | < $0.01 |
| Pinecone Storage | Free tier |
| Mistral LLM (per query) | ~$0.001-0.01 |

**Monthly estimate** (1000 queries): ~$1-10

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - feel free to use this project for learning or commercial purposes.

## 🙏 Acknowledgments

- **Mistral AI** - Embedding and LLM models
- **Pinecone** - Vector database
- **LangChain** - RAG framework
- **FastAPI** - API framework

## 📞 Support

For issues or questions:
1. Check the documentation in the guides
2. Review error messages carefully
3. Ensure API keys are set correctly
4. Verify all dependencies are installed

## 🎯 Roadmap

- [ ] Add user feedback mechanism
- [ ] Implement caching layer
- [ ] Add more credit cards
- [ ] Build web UI (React/Streamlit)
- [ ] Add authentication
- [ ] Implement analytics dashboard
- [ ] Add multi-language support
- [ ] Create mobile app

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ using Mistral AI, Pinecone, and LangChain**
