# Quick Start: Enhanced RAG

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt')"
```

### Step 2: Configure Enhanced RAG

```bash
# Copy example config
cp .env.example .env

# Edit .env
nano .env
```

**Minimal configuration for enhanced RAG:**

```bash
# API Key (choose one)
GEMINI_API_KEY=your_key_here
# OR
OPENAI_API_KEY=your_key_here

# Enhanced RAG Settings
CHUNKING_STRATEGY=section
USE_MMR_RETRIEVAL=true
EMBEDDING_STRATEGY=local
```

### Step 3: Verify Installation

```bash
python3 verify_installation.py
```

Should see: `✅ All checks passed!`

### Step 4: Start Server

```bash
python3 -m app.main
```

Server starts at: http://localhost:8000

---

## 📝 Usage Examples

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/absolute/path/to/policy.pdf"
  }'
```

**Response:**
```json
{
  "success": true,
  "sections_created": 15,
  "key_terms_learned": 40,
  "clusters_created": 6
}
```

### Query System

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the maternity leave policy?"
  }'
```

**Response with Enhanced Fields:**
```json
{
  "answer": "Employees are entitled to 16 weeks of maternity leave...",
  "sources": [...],
  "confidence_score": 0.92,
  "retrieval_method": "kg_guided",
  "mmr_sentences_used": 6,
  "section_title": "Maternity Leave",
  "precision_at_k": 0.85,
  "recall_at_k": 0.90,
  "mrr": 0.95
}
```

---

## ⚙️ Configuration Options

### Embedding Strategy

```bash
# Use provider-based (OpenAI/Gemini) - requires API key, higher cost
EMBEDDING_STRATEGY=provider

# Use local (SentenceTransformer) - free, no API calls
EMBEDDING_STRATEGY=local
```

### MMR Parameters

```bash
# Number of sentences to select (default: 6)
MMR_K=6

# Relevance vs diversity tradeoff (default: 0.7)
# Higher = more relevant, less diverse
# Lower = more diverse, less relevant
MMR_LAMBDA=0.7
```

### Clustering

```bash
# Number of semantic clusters (default: 6)
N_CLUSTERS=6
```

### LLM Refinement

```bash
# Join MMR sentences directly (fast, low cost)
USE_LLM_REFINEMENT=false

# Use LLM to refine answer (better quality, higher cost)
USE_LLM_REFINEMENT=true
```

---

## 🔄 Switch Between Strategies

### Use Enhanced RAG (Recommended)

```bash
CHUNKING_STRATEGY=section
USE_MMR_RETRIEVAL=true
```

**Features:**
- Section-based chunking
- Knowledge-guided retrieval
- MMR sentence selection
- Higher accuracy (P@3 > 0.80)

### Use Legacy RAG

```bash
CHUNKING_STRATEGY=fixed
USE_MMR_RETRIEVAL=false
```

**Features:**
- Fixed-size chunking (1000 chars)
- Direct vector search
- Simpler, faster setup
- Good for small documents

---

## 🧪 Run Tests

```bash
# All tests
pytest -v

# Specific tests
pytest tests/test_section_processor.py -v
pytest tests/test_mmr_retriever.py -v

# With coverage
pytest --cov=app --cov-report=html
```

---

## 📊 Expected Performance

| Metric | Legacy | Enhanced |
|--------|--------|----------|
| Precision@3 | 0.33 | 0.80+ |
| Recall@3 | 1.00 | 0.90+ |
| Query Time | ~300ms | <500ms |
| API Cost | High | Lower |

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "punkt not found" error
```bash
python3 -c "import nltk; nltk.download('punkt')"
```

### Settings not updating
```bash
# Check .env file exists
cat .env

# Verify settings loaded
python3 -c "from config.settings import settings; print(settings.chunking_strategy)"
```

### Enhanced strategy not activating
```bash
# Both must be true:
CHUNKING_STRATEGY=section
USE_MMR_RETRIEVAL=true

# Verify:
python3 verify_installation.py
```

---

## 📚 Documentation

- **README.md** - Full project documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical details
- **verify_installation.py** - Installation checker

---

## 💡 Pro Tips

1. **Start with local embeddings** to avoid API costs:
   ```bash
   EMBEDDING_STRATEGY=local
   ```

2. **Disable LLM refinement** for faster responses:
   ```bash
   USE_LLM_REFINEMENT=false
   ```

3. **Adjust MMR parameters** based on your needs:
   - More relevant: `MMR_LAMBDA=0.9`
   - More diverse: `MMR_LAMBDA=0.5`

4. **Monitor metrics** in the response:
   - `precision_at_k` - accuracy of retrieval
   - `retrieval_method` - which strategy was used

---

## 🎯 What Makes Enhanced RAG Better?

### Traditional RAG (Legacy)
```
PDF → Fixed chunks → Embed → Store → Query → Top-k chunks → LLM → Answer
```

### Enhanced RAG
```
PDF → Sections → Learn terms → Build KG → Cluster
                                              ↓
Query → Normalize → KG search → Fallback cluster → MMR select → Answer
```

**Key Improvements:**
- **Semantic sections** (not arbitrary splits)
- **Knowledge-guided** (not just vector similarity)
- **Diverse answers** (MMR reduces redundancy)
- **Explainable** (sentence-level attribution)

---

**Ready to try it?**

```bash
# 1. Configure
cp .env.example .env
# Edit .env with your API key

# 2. Verify
python3 verify_installation.py

# 3. Start
python3 -m app.main
```

🚀 **Happy querying!**
