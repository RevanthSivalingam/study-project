# Enterprise Policy Chatbot 🤖

**RAG + Knowledge Graph chatbot for enterprise policy documents**
*Simplified setup - No database server required!*

## 🎯 Overview

Ask questions about your policy documents and get accurate answers with source references.

**Example:**
- **Question**: "What's the maternity leave policy and how do I apply?"
- **Answer**: "The maternity leave policy provides 16 weeks of paid leave..."
- **Sources**: maternity_policy.pdf (Page 2, relevance: 0.92)

## ⚡ Key Features

- **Upload PDFs** - Automatically processes and indexes policy documents
- **Semantic Search** - Finds relevant information even with different wording
- **Knowledge Graph** - Extracts entities (policies, departments, benefits) and their relationships
- **Source References** - Every answer includes document name + page number
- **Crisp Answers** - GPT-4 generates concise, accurate responses
- **REST API** - Easy integration with FastAPI

## 🏗️ Architecture

```
PDF Document → Extract & Chunk → [ChromaDB Vectors] + [NetworkX Graph]
                                           ↓
User Question → Vector Search + Graph Query → GPT-4 → Answer + Sources
```

### Tech Stack

- **Backend**: Python + FastAPI
- **Vector DB**: ChromaDB (embeddings for semantic search)
- **Knowledge Graph**: NetworkX (in-memory graph, persists to disk)
- **RAG Framework**: LangChain
- **LLM**: OpenAI GPT-4

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **OpenAI API Key** - Get from: https://platform.openai.com/api-keys

That's it! No database servers to install.

### Installation

**1. Clone and Setup**

```bash
cd study-project

# Option A: Use the quick start script
./start.sh

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure Environment**

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

`.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
```

**3. Create Data Directory**

```bash
mkdir -p data/pdfs
```

**4. Start the Server**

```bash
python -m app.main
```

✅ **API running at**: http://localhost:8000
📚 **Interactive docs**: http://localhost:8000/docs

## 📖 Usage

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Upload Policy Documents

Place your PDFs in `data/pdfs/` and upload:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/absolute/path/to/data/pdfs/leave_policy.pdf"
  }'
```

**Response:**
```json
{
  "document_id": "a1b2c3d4e5f6",
  "file_name": "leave_policy.pdf",
  "status": "processed",
  "chunks_created": 15,
  "entities_extracted": 8,
  "message": "Document successfully processed and indexed"
}
```

### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vacation policy?"
  }'
```

**Response:**
```json
{
  "answer": "Full-time employees receive 20 days of vacation per year...",
  "sources": [
    {
      "document_name": "leave_policy.pdf",
      "page_number": 3,
      "chunk_id": "a1b2c3_chunk_5",
      "relevance_score": 0.94,
      "excerpt": "Vacation Policy: Full-time employees are entitled to..."
    }
  ],
  "confidence_score": 0.91,
  "entities_found": ["vacation", "leave", "benefits"]
}
```

### 4. Using the Example Script

```bash
# Interactive demo
python example_usage.py
```

## 📁 Project Structure

```
study-project/
├── app/
│   ├── api/
│   │   └── routes.py              # API endpoints
│   ├── services/
│   │   ├── document_processor.py  # PDF extraction & chunking
│   │   ├── vector_store.py        # ChromaDB operations
│   │   ├── knowledge_graph.py     # NetworkX graph (in-memory)
│   │   └── rag_service.py         # Main RAG pipeline
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   └── main.py                    # FastAPI app
├── config/
│   └── settings.py                # Configuration
├── data/
│   ├── pdfs/                      # Your PDF files
│   ├── chroma_db/                 # Vector embeddings
│   └── knowledge_graph.pkl        # Saved graph
├── requirements.txt               # Dependencies
├── .env.example                   # Config template
├── start.sh                       # Quick start script
└── example_usage.py               # Demo script
```

## 🔧 How It Works

### Document Upload Flow

1. **Extract**: Read PDF text page by page
2. **Chunk**: Split into ~1000 character segments
3. **Embed**: Generate vector embeddings (OpenAI)
4. **Store Vectors**: Save to ChromaDB
5. **Extract Entities**: Use GPT-4 to identify policies, departments, benefits
6. **Build Graph**: Store entities and relationships in NetworkX
7. **Persist**: Save graph to disk (knowledge_graph.pkl)

### Query Flow

1. **Vector Search**: Find top 4 most similar document chunks
2. **Graph Query**: Find related entities using BFS traversal
3. **Context Assembly**: Combine chunks + entity relationships
4. **Generate Answer**: GPT-4 creates response from context
5. **Return**: Answer + sources (document, page, relevance score)

## ⚙️ Configuration

### Adjust Chunk Size

Edit `config/settings.py`:
```python
chunk_size: int = 1500      # Larger chunks (default: 1000)
chunk_overlap: int = 300    # More overlap (default: 200)
```

### Change LLM Model

Edit `app/services/rag_service.py`:
```python
# Use cheaper model
self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Or use local model (requires Ollama)
from langchain_community.llms import Ollama
self.llm = Ollama(model="llama2")
```

### Knowledge Graph Settings

The graph is automatically saved to:
- Location: `data/chroma_db/knowledge_graph.pkl`
- Auto-loads on startup
- Auto-saves after each document processing

## 💡 Example Use Cases

### HR Policy Chatbot
- Upload: Leave policies, benefits handbook, code of conduct
- Questions: "How do I request parental leave?", "What's covered by health insurance?"

### Engineering Docs
- Upload: API documentation, architecture guides, deployment procedures
- Questions: "How do I deploy to production?", "What's the authentication flow?"

### Compliance & Legal
- Upload: Compliance documents, legal policies, regulatory guidelines
- Questions: "What are the data retention requirements?", "GDPR compliance steps?"

## 🎨 Advantages of NetworkX Approach

### ✅ Benefits
- **Simple Setup** - No database server to install/configure
- **Lightweight** - Runs entirely in-memory
- **Portable** - Graph saved as pickle file
- **Fast Development** - Quick to iterate and test
- **Cost Effective** - No DB hosting costs

### ⚠️ Limitations
- **Scale** - Best for <10,000 documents
- **Concurrency** - Single-process (OK for small teams)
- **Query Power** - Less sophisticated than Neo4j Cypher

For larger deployments (100K+ docs, multi-user), consider upgrading to Neo4j.

## 🔍 API Endpoints

### `GET /api/v1/health`
Check system health and stats

### `POST /api/v1/documents/upload`
Upload and process PDF document

**Body:**
```json
{
  "file_path": "/path/to/document.pdf",
  "document_type": "policy"
}
```

### `POST /api/v1/chat`
Query the chatbot

**Body:**
```json
{
  "query": "your question here",
  "session_id": "optional-session-id"
}
```

### `GET /api/v1/stats`
Get system statistics (document count, entity count, etc.)

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
```

### "OpenAI API key not found"
```bash
# Check .env file exists and has correct key
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

### "File not found" when uploading
- Use **absolute paths** for file uploads
- Example: `/Users/username/study-project/data/pdfs/file.pdf`

### Graph not persisting
- Check `data/chroma_db/` directory has write permissions
- Graph auto-saves after each document upload

### Out of memory
- Reduce `chunk_size` in `config/settings.py`
- Process fewer documents at once
- For large datasets, consider upgrading to Neo4j

## 🚀 Future Enhancements

- [ ] Add frontend UI (React/Vue)
- [ ] Support Word docs, Excel, etc.
- [ ] Conversation history per session
- [ ] Cloud storage integration (S3, Google Drive)
- [ ] Multi-user sessions
- [ ] Caching for frequent questions
- [ ] Export graph to visualization (GraphViz, D3.js)
- [ ] Advanced entity extraction with custom NER
- [ ] Upgrade to Neo4j for large-scale deployment

## 📝 Development Notes

### Run in Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### View Graph Contents

```python
import pickle
import networkx as nx

# Load graph
with open('data/chroma_db/knowledge_graph.pkl', 'rb') as f:
    graph = pickle.load(f)

# View stats
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# List all entities
for node, data in graph.nodes(data=True):
    if data.get('node_type') == 'entity':
        print(f"{data['name']} ({data['entity_type']})")
```

### Reset Everything

```bash
# Clear all data
rm -rf data/chroma_db/
mkdir -p data/chroma_db

# Restart server
python -m app.main
```

## 📄 License

MIT

## 🤝 Contributing

Issues and pull requests welcome!

## 💬 Support

Questions? Open an issue on the repository.

---

**Built with Claude Code** 🚀
