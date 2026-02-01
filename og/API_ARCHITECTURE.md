# 🏗️ API Architecture Guide

## Overview

Your RAG Chatbot now has a **full backend API architecture** similar to the reference project, with separate backend and frontend services.

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      USER BROWSER                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│              STREAMLIT UI (Frontend)                     │
│              Port: 8501                                  │
│              File: ui_api.py                             │
│  • Document upload interface                             │
│  • Chat interface                                        │
│  • Statistics display                                    │
│  • Makes HTTP API calls                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ HTTP REST API
                     │ (POST/GET requests)
                     ↓
┌─────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (API Server)                │
│              Port: 8000                                  │
│              File: app/main.py                           │
│  • REST API endpoints                                    │
│  • Request validation (Pydantic)                         │
│  • RAG system management                                 │
│  • CORS enabled                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│              RAG SYSTEM (Core Logic)                     │
│              File: rag_logic.py                          │
│  • Document loading                                      │
│  • Embedding generation                                  │
│  • Knowledge graph                                       │
│  • Query processing                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "initialized": true
}
```

---

#### 2. Initialize System
```http
POST /api/v1/initialize
```

**Request Body:**
```json
{
  "api_key": "optional-openai-key",
  "input_folder": "inputfiles"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully initialized with 1 documents and 45 sections",
  "stats": {
    "documents": 1,
    "sections": 45,
    "clusters": 6,
    "learned_terms": 40,
    "kg_entities": 120
  }
}
```

---

#### 3. Upload Document
```http
POST /api/v1/documents/upload
```

**Request Body:**
```json
{
  "file_path": "/path/to/document.pdf",
  "document_type": "policy",
  "metadata": {
    "category": "HR",
    "department": "Human Resources"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully initialized with 1 documents and 45 sections",
  "document_id": "document",
  "chunks_created": 45,
  "entities_extracted": 120
}
```

---

#### 4. Chat Query
```http
POST /api/v1/chat
```

**Request Body:**
```json
{
  "query": "What is the maternity leave policy?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "MATERNITY LEAVE POLICY 4.1 Eligibility...",
  "sources": [
    {
      "document_name": "PAGE 4 - MATERNITY LEAVE POLICY",
      "page_number": "N/A",
      "relevance_score": 0.85,
      "excerpt": "Female employees who have completed 6 months..."
    }
  ],
  "confidence_score": 0.8,
  "entities_found": [],
  "method": "knowledge_graph",
  "section_title": "PAGE 4 - MATERNITY LEAVE POLICY"
}
```

---

#### 5. Get Statistics
```http
GET /api/v1/stats
```

**Response:**
```json
{
  "data": {
    "total_documents_indexed": 1,
    "total_entities": 120,
    "total_sections": 45,
    "total_clusters": 6,
    "learned_terms": 40
  }
}
```

---

#### 6. Reset System
```http
POST /api/v1/reset
```

**Response:**
```json
{
  "success": true,
  "message": "System reset successfully"
}
```

---

## 🚀 Startup Options

### Option 1: Start Both Services Together (Recommended)

```bash
./start_all.sh
```

**Requires:** `tmux` (install with `brew install tmux` on macOS)

**What it does:**
- Starts backend on port 8000
- Starts frontend on port 8501
- Both run in a tmux session

---

### Option 2: Manual Startup (Two Terminals)

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
# Wait 5 seconds for backend to start
./start_frontend.sh
```

---

### Option 3: Python Direct

**Terminal 1 - Backend:**
```bash
python3 -m app.main
```

**Terminal 2 - Frontend:**
```bash
streamlit run ui_api.py
```

---

## 📊 File Structure

```
/Users/rsivalingam/workspace/simple/
├── app/
│   ├── __init__.py            # Package init
│   └── main.py                # FastAPI backend server
├── rag_logic.py               # Core RAG system
├── ui_api.py                  # Streamlit UI (API version)
├── ui.py                      # Streamlit UI (standalone version)
├── start_backend.sh           # Backend startup script
├── start_frontend.sh          # Frontend startup script
├── start_all.sh               # Start both services
├── requirements.txt           # Python dependencies
└── inputfiles/                # Document storage
```

---

## 🔄 Request Flow

### Document Upload Flow

```
User uploads PDF in UI
        ↓
UI saves file to inputfiles/
        ↓
UI calls POST /api/v1/documents/upload
        ↓
Backend receives request
        ↓
Backend initializes RAG system
        ↓
RAG loads document, creates embeddings
        ↓
Backend returns stats (sections, entities)
        ↓
UI displays success message
```

### Query Flow

```
User types question
        ↓
UI calls POST /api/v1/chat
        ↓
Backend receives query
        ↓
RAG system processes:
  1. Knowledge Graph search
  2. Semantic search (fallback)
  3. MMR sentence selection
        ↓
Backend formats response
        ↓
UI displays answer + sources
```

---

## 🛠️ Development

### Testing Backend API

**Using curl:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get stats
curl http://localhost:8000/api/v1/stats

# Chat query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?"}'
```

**Using Browser:**
- API Docs: http://localhost:8000/docs
- Interactive testing with Swagger UI

---

### Testing Frontend

1. Start backend first
2. Start frontend
3. Upload documents via UI
4. Ask questions

---

## 🔐 Security Considerations

### Current Setup (Development)
- ✅ CORS enabled for all origins
- ✅ No authentication required
- ✅ Local network only

### Production Recommendations
- 🔒 Add authentication (JWT tokens)
- 🔒 Restrict CORS to specific origins
- 🔒 Use HTTPS
- 🔒 Add rate limiting
- 🔒 Validate file uploads
- 🔒 Sanitize user inputs

---

## 📈 Performance

### Backend (FastAPI)
- **Startup:** ~2-5 seconds
- **First query:** ~30 seconds (model loading)
- **Subsequent queries:** ~1-3 seconds
- **Concurrent requests:** Async-capable

### Frontend (Streamlit)
- **Startup:** ~3-5 seconds
- **API calls:** ~1-3 seconds
- **UI updates:** Real-time

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Frontend Can't Connect

**Error:** "API Offline"

**Check:**
1. Is backend running? `curl http://localhost:8000/api/v1/health`
2. Correct port? Backend should be on 8000
3. Firewall blocking? Check network settings

**Solution:**
```bash
# Restart backend
./start_backend.sh
```

---

### CORS Errors

**Error:** "CORS policy blocked"

**Check:** `app/main.py` line 20-27
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be ["*"] for development
    ...
)
```

---

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or port 8501
lsof -ti:8501 | xargs kill -9
```

---

## 🔄 Architecture Comparison

### Standalone (ui.py)
```
UI → Direct Import → RAG Logic
```
**Pros:**
- Simple setup
- Single process
- No API overhead

**Cons:**
- Tight coupling
- No API for other clients
- Can't scale independently

### API-Based (ui_api.py + app/main.py)
```
UI → HTTP API → Backend → RAG Logic
```
**Pros:**
- Loose coupling
- API available for other clients
- Scale independently
- Better for production

**Cons:**
- More complex setup
- Network overhead
- Requires two processes

---

## 🎯 When to Use Which

### Use Standalone (`ui.py`)
- Quick testing
- Development
- Single user
- Simple deployment

### Use API-Based (`ui_api.py` + `app/main.py`)
- Production deployment
- Multiple clients (web, mobile, etc.)
- Need API documentation
- Team collaboration
- Microservices architecture

---

## 📚 API Documentation

**Interactive Docs:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Features:**
- Try out endpoints directly
- See request/response schemas
- View all available endpoints
- Generate client code

---

## 🔮 Future Enhancements

**Planned API Features:**
- [ ] Authentication & authorization
- [ ] User management
- [ ] Document versioning
- [ ] Batch processing
- [ ] Webhook notifications
- [ ] Rate limiting
- [ ] Caching layer
- [ ] Async document processing
- [ ] WebSocket for real-time updates

---

## ✅ Checklist

Before deploying:
- [ ] Backend starts successfully
- [ ] Frontend starts successfully
- [ ] Can upload documents
- [ ] Can query documents
- [ ] Stats display correctly
- [ ] Error handling works
- [ ] API documentation accessible
- [ ] Both services restart cleanly

---

**Architecture Status:** ✅ Complete & Ready

**Last Updated:** February 1, 2026
