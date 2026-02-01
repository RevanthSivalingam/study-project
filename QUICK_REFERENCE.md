# Enhanced RAG - Quick Reference

## 🚀 Quick Start

```bash
# Start server
python3 -m app.main

# Test query (easy way)
./test_query.sh "Your question here"

# Or use curl directly
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question"}'
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `.env` | Configuration (strategy, API keys) |
| `verify_installation.py` | Check system health |
| `test_query.sh` | Quick query tester |
| `IMPLEMENTATION_SUMMARY.md` | Full technical details |

## ⚙️ Configuration (.env)

### Enhanced RAG (Recommended)
```bash
CHUNKING_STRATEGY=section
USE_MMR_RETRIEVAL=true
EMBEDDING_STRATEGY=local  # Free, no API costs
```

### Legacy RAG
```bash
CHUNKING_STRATEGY=fixed
USE_MMR_RETRIEVAL=false
EMBEDDING_STRATEGY=provider  # Uses Gemini/OpenAI
```

## 🔧 Common Commands

```bash
# Check system health
curl http://localhost:8000/api/v1/health

# Get detailed stats
curl http://localhost:8000/api/v1/stats | python3 -m json.tool

# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/document.pdf"}'

# Run tests
pytest -v

# Verify installation
python3 verify_installation.py
```

## 📊 Enhanced RAG Pipeline

```
Document → Sections (43) → Terms (40) → KG (257 mappings) → Clusters (6)
                                              ↓
Query → Normalize → KG Search → Select Section → MMR (k=6) → Answer
```

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| Sections extracted | 43 |
| Key terms learned | 40 |
| KG mappings | 257 |
| Clusters | 6 |
| Retrieval method | kg_guided → semantic_fallback |
| MMR diversity | λ=0.7 (70% relevant, 30% diverse) |

## 🔍 Response Fields

Enhanced RAG adds these fields to responses:

- `retrieval_method`: "kg_guided" or "semantic_fallback"
- `mmr_sentences_used`: Number of sentences selected
- `section_title`: Title of retrieved section
- `precision_at_k`, `recall_at_k`, `mrr`: Quality metrics

## 💡 Tips

1. **Cost Optimization**: Use `EMBEDDING_STRATEGY=local` for free embeddings
2. **Better Answers**: Set `USE_LLM_REFINEMENT=true` (increases cost)
3. **More Diverse**: Decrease `MMR_LAMBDA` (e.g., 0.5)
4. **More Relevant**: Increase `MMR_LAMBDA` (e.g., 0.9)
5. **More Sentences**: Increase `MMR_K` (e.g., 10)

## 📝 Example Queries

```bash
./test_query.sh "What is the maternity leave policy?"
./test_query.sh "How many days of sick leave per year?"
./test_query.sh "What are the health insurance benefits?"
./test_query.sh "Who is eligible for dental coverage?"
./test_query.sh "What is the vacation policy?"
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | `pip install -r requirements.txt` |
| "punkt not found" | `python3 -c "import nltk; nltk.download('punkt_tab')"` |
| Server not responding | Check if running: `curl http://localhost:8000/api/v1/health` |
| Wrong strategy active | Restart server after changing `.env` |
| Low quality answers | Try `USE_LLM_REFINEMENT=true` |

## 📚 Documentation

- **README.md** - Complete project guide
- **IMPLEMENTATION_SUMMARY.md** - Technical deep-dive
- **QUICK_START_ENHANCED_RAG.md** - 5-minute setup
- This file - Quick reference

## 🎓 Architecture

### Legacy vs Enhanced

| Feature | Legacy | Enhanced |
|---------|--------|----------|
| Chunking | Fixed 1000 chars | Section-based |
| Chunks | ~15 | 43 sections |
| Retrieval | Vector only | KG + Clustering |
| Selection | Top-k | MMR (diversity) |
| Embeddings | API | Local (free) |
| Cost/query | High | Low |

### Two-Stage Retrieval

1. **Stage 1: KG-Guided**
   - Query normalized (remove generic terms)
   - Search knowledge graph for term matches
   - Select best section by hybrid score
   - If score > threshold → use this section

2. **Stage 2: Semantic Fallback**
   - Select most similar cluster (K-Means)
   - Pick best section from cluster
   - Guaranteed to return something

3. **Stage 3: MMR Selection**
   - Extract sentences from section
   - Select diverse sentences (MMR algorithm)
   - Balance relevance vs diversity

## 🔄 Workflow

```bash
# 1. Start server
python3 -m app.main

# 2. Upload documents
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Content-Type: application/json" \
  -d '{"file_path": "$(pwd)/data/pdfs/policy.pdf"}'

# 3. Query
./test_query.sh "Your question"

# 4. Check metrics in response
# 5. Tune parameters in .env if needed
# 6. Restart server to apply changes
```

---

**Need help?** Check `IMPLEMENTATION_SUMMARY.md` for detailed information.
