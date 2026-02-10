# RAG System - Quick Start Guide

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install google-generativeai==0.3.2 python-dotenv==1.0.0
```

---

## 🎯 Three Ways to Use

### 1️⃣ No LLM (Retrieval-Only Mode - Default)

**Best for**: Quick testing, no API costs, basic queries

```bash
python3 -m streamlit run ui.py
```

**What you get**:
- FREE - no API costs
- Fast responses
- **Precise retrieval** (4-5 most relevant sentences)
- Direct concatenation
- Real confidence scores
- Source attribution with relevance

**Example Output** (beautifully formatted):
```markdown
**PERFORMANCE REVIEW CYCLE**

**Annual Performance Review:**

Performance reviews are conducted annually. Employees receive feedback from managers.

The review evaluates:
- Goal achievement
- Professional development
- Future objectives

🎯 Confidence: 🟡 72% (High)
Method: 🔍 Semantic Search
🤖 Provider: MMR

📄 Source Section: Performance Reviews

📎 View 4 Sources ▼
  Source #1    Relevance: 🟢 Very High (85%)
  Source #2    Relevance: 🟡 High (78%)
```

---

### 2️⃣ With OpenAI (Smart Synthesis Mode)

**Best for**: Natural language answers, complex queries, best quality

**Step 1: Configure**
```bash
cp .env.example .env
nano .env
```

Set in `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_MAX_TOKENS=500
OPENAI_TEMPERATURE=0.3
```

**Step 2: Run**
```bash
python3 -m streamlit run ui.py
```

**Step 3: In UI**
- Select "openai" from dropdown
- Enter API key (or reads from .env)
- Upload documents
- Ask questions

**What you get**:
- Natural language synthesis
- **Comprehensive retrieval** (8-10 relevant sentences)
- LLM extracts and combines information
- Higher quality answers
- Real confidence scores
- Token usage tracking
- ~$0.01 per query (GPT-4)

**Example Output** (beautifully formatted):
```markdown
**PERFORMANCE REVIEW CYCLE**

**Annual Performance Review:**

According to company policy, performance reviews are conducted annually where employees receive structured feedback from their managers.

**Evaluation Focus:**
- Goal achievement
- Skill development
- Future objectives

🎯 Confidence: 🟢 89% (Very High)
Method: 📊 Knowledge Graph
🤖 Provider: OPENAI
💰 Tokens Used: 245

📄 Source Section: Performance Reviews

📎 View 8 Sources ▼
  Source #1    Relevance: 🟢 Very High (92%)
  Source #2    Relevance: 🟢 Very High (88%)
```

---

### 3️⃣ With Google Gemini (Cost-Effective Synthesis)

**Best for**: Budget-conscious, good quality, free tier available

**Step 1: Configure**
```bash
nano .env
```

Set in `.env`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL=gemini-1.5-pro
GEMINI_MAX_TOKENS=500
GEMINI_TEMPERATURE=0.3
```

**Step 2: Run**
```bash
python3 -m streamlit run ui.py
```

**Step 3: In UI**
- Select "gemini" from dropdown
- Enter Gemini API key
- Upload documents
- Ask questions

**What you get**:
- Natural language synthesis
- **Comprehensive retrieval** (8-10 relevant sentences)
- Good quality answers
- FREE tier: 50 queries/day
- Paid: ~$0.005 per query (50% cheaper than OpenAI)

---

## 🔧 API Mode (Backend + Frontend)

**For production deployments**

**Terminal 1 - Backend**:
```bash
cd /path/to/og
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd /path/to/og
streamlit run ui_api.py --server.port 8501
```

**Access**: http://localhost:8501

---

## 📊 Understanding the Output

### **Confidence Indicator**
```
🟢 ≥85% - Very High: Trust this answer
🟡 70-84% - High: Good answer quality
🟠 55-69% - Medium: Verify with sources
🔴 <55% - Low: May be incomplete
```

### **Retrieval Method**
```
📊 Knowledge Graph: Found exact term matches (more reliable)
🔍 Semantic Search: Found semantic matches (good fallback)
```

### **Source Relevance**
```
🟢 Very High (≥80%): Highly relevant
🟡 High (65-79%): Relevant
🟠 Medium (50-64%): Somewhat relevant
🔴 Low (<50%): Tangentially relevant
```

---

## ⚙️ Configuration Guide

### **Performance vs Quality Tradeoffs**

```env
# Balanced (Recommended)
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

# More Precise (for factual queries)
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=300

# More Creative (for explanations)
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=700
```

### **Cost Control**

```env
# Maximum cost control
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper
LLM_MAX_TOKENS=300          # Shorter responses

# Best quality
OPENAI_MODEL=gpt-4-turbo-preview
LLM_MAX_TOKENS=700
```

---

## 🔍 Troubleshooting

### **Low Confidence Scores (<55%)**

**Causes**:
- Query doesn't match document content well
- Information genuinely not in documents
- Generic query terms

**Solutions**:
- Rephrase query to match document terminology
- Be more specific in your question
- Check if information exists in uploaded documents

### **"Information not available"**

**Causes**:
- No relevant sections found
- Query too vague or off-topic

**Solutions**:
- Upload more relevant documents
- Rephrase with keywords from your documents
- Try broader query terms

### **LLM Fallback Warning**

```
⚠️ LLM unavailable, using retrieval-only mode
```

**Causes**:
- Invalid API key
- Rate limit exceeded
- Network issue
- API service down

**Auto-Fallback**:
- System automatically switches to MMR mode
- You still get answers
- Check API key and service status

---

## 💰 Cost Estimation

### **OpenAI (GPT-4 Turbo)**
- Input: ~$0.01 per 1K tokens
- Output: ~$0.03 per 1K tokens
- **Average query**: $0.008-$0.015
- **100 queries**: ~$1.00

### **Google Gemini 1.5 Pro**
- **Free tier**: 50 queries/day
- **Paid**: ~$0.005 per query
- **100 queries**: ~$0.50

### **MMR (No LLM)**
- **FREE** - Zero API costs
- Only local compute
- Unlimited queries

---

## 🎓 Example Workflow

### **1. Start the System**
```bash
python3 -m streamlit run ui.py
```

### **2. Choose Your Mode**

**For Testing**: Select "none" (MMR mode)
**For Production**: Select "openai" or "gemini"

### **3. Upload Documents**
- Click "Browse files"
- Select PDF/TXT/JSON files
- Click "Upload & Initialize"
- Wait for "Successfully initialized..."

### **4. Ask Questions**

**Good questions**:
- "How many vacation days do employees get?"
- "What is the remote work policy?"
- "When are performance reviews conducted?"

**Tips**:
- Use terminology from your documents
- Be specific but not overly narrow
- One question at a time for best results

### **5. Interpret Results**

**Check Confidence**:
- 🟢 Very High/High: Trust the answer
- 🟠 Medium: Review sources
- 🔴 Low: Rephrase or verify

**Review Sources**:
- Click "📎 View Sources"
- Check relevance scores
- Read actual text excerpts
- Verify information is accurate

---

## 🎯 Best Practices

### **Document Preparation**
✅ Clear section headers
✅ Well-structured content
✅ Consistent terminology
✅ PDF, TXT, or JSON format

### **Query Writing**
✅ Use specific terms from documents
✅ Ask one thing at a time
✅ Include key nouns/verbs
✅ Avoid overly generic questions

### **Result Interpretation**
✅ Always check confidence score
✅ Review high-relevance sources
✅ Cross-reference multiple sources
✅ Verify critical information

---

## 📚 Learn More

- **Full documentation**: `README.md`
- **Configuration options**: `.env.example`
- **Code documentation**: `rag_logic.py`, `llm_provider.py`

---

## 🆘 Need Help?

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **No Documents Found**
   - Check `inputfiles/` folder exists
   - Verify file formats (PDF/TXT/JSON)
   - Ensure files have content

3. **Low Confidence on All Queries**
   - Documents may not match query domain
   - Try more specific questions
   - Verify document quality

4. **API Key Invalid**
   - Check key format in `.env`
   - Verify key is active
   - Check API service status

---

## ✨ New Features

### **LLM-Aware Retrieval**
- Automatically adapts to available LLM
- More context with LLM (8-10 sentences)
- Precise retrieval without LLM (4-5 sentences)

### **Real Confidence Scores**
- Actual similarity-based scores
- Not placeholders!
- Multiple factors considered
- Color-coded for quick assessment

### **Source Attribution**
- Every sentence tracked
- Real relevance scores shown
- Section information included
- Ranking by relevance

### **Beautiful Answer Formatting**
- Unified beautification for both MMR and LLM modes
- Bold section headers: **2.1 Title:**
- Bold key terms: **required**, **mandatory**, **eligible**
- Automatic bullet points for lists
- Paragraph breaks for easy scanning
- Scannable, professional layout

---

**Ready to go!** 🎉

Start with: `python3 -m streamlit run ui.py`

For best results, use **OpenAI** or **Gemini** for natural language synthesis!
