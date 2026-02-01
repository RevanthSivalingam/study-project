# 🚀 Quick Start Guide

Get your RAG chatbot running in 3 simple steps!

## Step 1: Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

## Step 2: Add Documents

Place your PDF, TXT, or JSON files in the `inputfiles/` folder:

```bash
# Example: Copy your policy documents
cp /path/to/your/policies/*.pdf inputfiles/
```

## Step 3: Start the UI

```bash
# Make script executable (first time only)
chmod +x start_ui.sh

# Start the application
./start_ui.sh
```

## 🎯 Using the Application

### Upload Documents via UI
1. Open **http://localhost:8501** in your browser
2. Click **"Choose PDF/TXT/JSON files"** in the sidebar
3. Select files and click **"📤 Upload & Process"**
4. Wait for initialization

### Ask Questions
1. Type your question in the chat input
2. Press Enter
3. View answer with source citations

## 💡 Example Workflow

```
1. Upload: "employee_handbook.pdf"
2. Wait: "Successfully initialized with 1 documents and 45 sections"
3. Ask: "What is the vacation policy?"
4. Get: Answer with relevant sections and confidence
```

## 🔧 Troubleshooting

### Problem: Command not found
```bash
# Make sure you're in the project directory
cd /Users/rsivalingam/workspace/simple

# Make script executable
chmod +x start_ui.sh
```

### Problem: Module not found
```bash
# Install dependencies
pip install -r requirements.txt
```

### Problem: NLTK data missing
```bash
# Run setup again
python setup.py
```

## 📊 System Check

After starting, verify in the sidebar:
- ✅ Documents Loaded: > 0
- ✅ Sections Extracted: > 0
- ✅ Knowledge Graph Entities: > 0

## 🎉 Ready to Go!

You're all set! Start asking questions about your documents.

**Example Questions:**
- "What is the leave policy?"
- "How many vacation days?"
- "What are the benefits?"
- "Tell me about maternity leave"

---

**Need help?** Check the full README.md for detailed documentation.
