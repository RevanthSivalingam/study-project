# RAG Policy Chatbot - Simple UI

A knowledge-guided Retrieval-Augmented Generation (RAG) system with an intuitive Streamlit interface for document-based question answering.

## 🎯 Features

### Core RAG Capabilities
- **Document Loading**: PDF, TXT, and JSON support
- **Intelligent Chunking**: Section-based text segmentation
- **Semantic Embeddings**: Using all-mpnet-base-v2 model
- **Knowledge Graph**: Automatic entity extraction and relationship mapping
- **Clustering**: K-means clustering for topic organization
- **MMR Retrieval**: Maximal Marginal Relevance for diverse results
- **Dual Retrieval**: Knowledge Graph + Semantic Search fallback

### UI Features
- **Document Upload**: Drag-and-drop interface for multiple files
- **Real-time Processing**: Instant document indexing
- **Interactive Chat**: Conversational Q&A interface
- **Source Citations**: View retrieved sentences for each answer
- **Retrieval Method Display**: See whether KG or semantic search was used
- **System Statistics**: Track documents, sections, and entities
- **Session Management**: Clear chat or reset entire system

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

### 3. Start the UI

```bash
./start_ui.sh
```

Or manually:
```bash
streamlit run ui.py
```

The UI will open at: **http://localhost:8501**

## 📖 How to Use

### Step 1: Upload Documents
1. Click **"Choose PDF/TXT/JSON files"** in the sidebar
2. Select one or more documents
3. Click **"📤 Upload & Process"**
4. Wait for initialization to complete

### Step 2: Ask Questions
1. Type your question in the chat input
2. Press Enter to submit
3. View the answer with:
   - **Retrieved sentences** from relevant sections
   - **Retrieval method** used (Knowledge Graph or Semantic Search)
   - **Source section** that answered your question

### Step 3: Explore Results
- Click **"📎 View Retrieved Sentences"** to see exact text snippets
- Check **System Stats** to see what's been processed
- Use **Clear Chat History** to start fresh

## 🎨 UI Components

### Sidebar
- **Configuration**: Optional OpenAI API key
- **Document Upload**: Multi-file upload with progress
- **System Stats**: Real-time metrics
- **Actions**: Initialize, clear chat, clear all data

### Main Chat Area
- **Message History**: Full conversation with metadata
- **Source Citations**: Expandable sentence view
- **Method Indicators**:
  - 🟢 **Knowledge Graph** (structured retrieval)
  - 🔵 **Semantic Search** (embedding-based)

## 📊 Example Questions

```
❓ What is the maternity leave policy?
❓ How many vacation days do employees get?
❓ What is the process for requesting time off?
❓ Tell me about the sick leave policy
❓ What are the eligibility criteria for benefits?
```

## 🧠 How It Works

### Thought Process

**1. Document Processing**
- Load documents from inputfiles folder
- Extract sections using header detection
- Generate embeddings for each section

**2. Knowledge Building**
- Learn key terms using TF-IDF
- Build knowledge graph with entities and relationships
- Cluster sections by semantic similarity
- Identify generic terms to filter out

**3. Query Answering**
- **Step 1**: Try Knowledge Graph retrieval
  - Match query terms to entities
  - Find relevant sections
  - Use structured knowledge
- **Step 2**: Fallback to Semantic Search
  - Select most relevant cluster
  - Find best matching section
  - Use embedding similarity

**4. Answer Generation**
- Extract sentences from selected section
- Apply MMR for diverse sentence selection
- Return top-k relevant sentences

## 🔧 Configuration

### OpenAI API Key (Optional)
- Enter in the sidebar
- Currently used for future LLM integration
- System works in retrieval-only mode without it

### Customization Options

Edit `rag_logic.py` to adjust:
- `NUM_CLUSTERS = 6` - Number of topic clusters
- `top_percent=0.15` - Generic term threshold
- `top_k=40` - Number of key terms to learn
- `k=6` - MMR sentence count
- `lambda_param=0.7` - MMR relevance vs diversity

## 📁 Project Structure

```
simple/
├── ui.py                 # Streamlit interface
├── rag_logic.py          # Core RAG system
├── requirements.txt      # Python dependencies
├── start_ui.sh           # Startup script
├── inputfiles/           # Document storage
├── midsemcode.ipynb      # Original notebook
└── README.md             # This file
```

## 🔍 System Statistics Explained

- **Documents Loaded**: Total PDFs/TXT/JSON files processed
- **Sections Extracted**: Number of document sections identified
- **Clusters Created**: Topic groups for organization
- **Key Terms Learned**: Important domain-specific terms
- **Knowledge Graph Entities**: Structured entities extracted

## 🎯 Retrieval Methods

### 📊 Knowledge Graph (Primary)
- **Fast**: Uses structured entity matching
- **Accurate**: Based on learned domain knowledge
- **Explainable**: Clear entity → fact mappings

### 🔍 Semantic Search (Fallback)
- **Flexible**: Handles queries outside KG coverage
- **Robust**: Uses embedding similarity
- **Comprehensive**: Searches entire corpus

## 🛠️ Troubleshooting

### Issue: "System not initialized"
**Solution**: Upload documents and click "Upload & Process"

### Issue: No documents found
**Solution**: Ensure PDFs/TXT/JSON are in `inputfiles/` folder

### Issue: Poor answers
**Try**:
- Upload more relevant documents
- Be more specific in your questions
- Check if the information exists in uploaded docs

### Issue: Slow initialization
**Expected**: First run downloads embedding model (~400MB)
- Subsequent runs are much faster
- Model is cached locally

## 🚀 Advanced Usage

### Pre-load Documents
Place files in `inputfiles/` before starting:
```bash
cp /path/to/policies/*.pdf inputfiles/
./start_ui.sh
```

### Batch Processing
Upload multiple documents at once for comprehensive coverage

### Session Management
- Use "Clear Chat History" to reset conversation
- Use "Clear All Data" to remove all documents and start over

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (for embedding model)
- Internet connection (first run only, for model download)

## 🎓 Learning Resources

### Understanding the System
1. **Section Chunking**: Splits docs by headers
2. **TF-IDF**: Identifies important vs generic terms
3. **Embeddings**: Converts text to semantic vectors
4. **Cosine Similarity**: Measures text relatedness
5. **K-Means**: Groups similar sections
6. **MMR**: Selects diverse relevant results

### Key Concepts
- **RAG**: Retrieval-Augmented Generation
- **Knowledge Graph**: Structured entity relationships
- **Semantic Search**: Meaning-based retrieval
- **MMR**: Balances relevance and diversity

## 📞 Support

For issues or questions:
1. Check system stats to verify initialization
2. Review retrieved sentences to understand answers
3. Try different phrasings of your question
4. Ensure documents contain relevant information

## 🔮 Future Enhancements

- ✅ Document upload via UI
- ✅ Real-time statistics
- ✅ Source citations
- 🔄 LLM-based answer refinement
- 🔄 Multi-turn conversation context
- 🔄 Document versioning
- 🔄 Advanced filters (date, category)

---

**Built with**: Streamlit • Sentence Transformers • scikit-learn • NLTK
