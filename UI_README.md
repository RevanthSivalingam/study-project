# Enterprise Policy Chatbot - UI Guide

## Overview
Simple and intuitive web interface for interacting with the Enterprise Policy Chatbot.

## Features

### 📄 Document Management
- **Upload PDF documents** directly through the UI
- **Automatic processing** - extracts text, creates embeddings, and builds knowledge graph
- **Metadata support** - add category and department information
- **Real-time feedback** - see chunks created and entities extracted

### 💬 Chat Interface
- **Interactive chat** with your policy documents
- **Source citations** - every answer shows which documents were used
- **Confidence scores** - color-coded confidence levels (High: Green, Medium: Orange, Low: Red)
- **Entity highlighting** - shows key entities found in the conversation
- **Chat history** - maintains conversation context within session

### 📊 System Statistics
- **Documents indexed** - total number of documents in the system
- **Entities extracted** - total entities in the knowledge graph
- **Provider info** - shows which LLM provider is active (Gemini or OpenAI)

## Getting Started

### 1. Start the API Server

First, ensure the FastAPI backend is running:

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python -m app.main
```

The API will be available at `http://localhost:8000`

### 2. Install UI Dependencies

```bash
# Make sure streamlit is installed
pip install streamlit==1.29.0
```

### 3. Start the UI

In a **new terminal window**:

```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit UI
streamlit run ui.py
```

The UI will automatically open in your browser at `http://localhost:8501`

## Usage Guide

### Uploading Documents

1. Click **"Choose a PDF file"** in the sidebar
2. Select a PDF document from your computer
3. Choose the **Document Type** (policy, procedure, etc.)
4. (Optional) Add **Category** and **Department** metadata
5. Click **"📤 Upload & Process"**
6. Wait for processing to complete (you'll see chunks and entities count)

### Asking Questions

1. Type your question in the chat input at the bottom
2. Press **Enter** or click **Send**
3. View the answer with:
   - **Main response** in the chat
   - **Sources** - click "📎 View Sources" to see which documents were used
   - **Confidence score** - how confident the system is in its answer
   - **Entities** - key concepts identified in the conversation

### Example Questions

```
❓ What is the annual leave policy?
❓ How many sick days are employees entitled to?
❓ What is the process for requesting time off?
❓ Tell me about the maternity leave policy
❓ What are the working hours?
```

## UI Features Explained

### Confidence Score Colors

- **🟢 Green (80%+)**: High confidence - answer is well-supported by documents
- **🟠 Orange (60-79%)**: Medium confidence - answer has some support
- **🔴 Red (<60%)**: Low confidence - answer may need verification

### Source Cards

Each source shows:
- **Document name** - which file the information came from
- **Page number** - where in the document
- **Relevance score** - how relevant this source is to your question
- **Excerpt** - snippet of text from the source

### System Stats

Real-time statistics showing:
- Total documents indexed in the system
- Total entities in the knowledge graph
- Current LLM provider (Gemini or OpenAI)

## Troubleshooting

### "API Offline" Error

**Problem**: The UI shows "❌ API Offline"

**Solution**:
1. Make sure the FastAPI server is running on port 8000
2. Check that you've started it with: `python -m app.main`
3. Verify at `http://localhost:8000/docs`

### Document Upload Fails

**Problem**: Document upload shows an error

**Possible causes**:
1. **File not found** - make sure the PDF is valid
2. **Processing error** - check if the PDF is readable (not scanned/image-only)
3. **API key issue** - verify your Gemini or OpenAI API key is set in `.env`

### Chat Returns Low Confidence

**Problem**: Answers have low confidence scores

**Possible causes**:
1. **No relevant documents** - upload more documents related to the question
2. **Question too vague** - try being more specific
3. **Limited context** - the system only knows what's in uploaded documents

### Provider Mismatch

**Problem**: UI shows wrong provider

**Solution**:
1. Check your `.env` file
2. If `GEMINI_API_KEY` is set → uses Gemini
3. If only `OPENAI_API_KEY` is set → uses OpenAI
4. Restart both API server and UI after changing `.env`

## Keyboard Shortcuts

- **Enter** - Send message
- **Shift + Enter** - New line in input
- **Escape** - Clear input field

## Running Both Services Together

Use two terminal windows:

**Terminal 1 (API Server)**:
```bash
source venv/bin/activate
python -m app.main
```

**Terminal 2 (UI)**:
```bash
source venv/bin/activate
streamlit run ui.py
```

## Customization

### Changing UI Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1E88E5"      # Blue - change to your brand color
backgroundColor = "#FFFFFF"    # White background
secondaryBackgroundColor = "#F0F2F6"  # Light gray
textColor = "#262730"          # Dark text
```

### Changing Port

Edit `.streamlit/config.toml`:
```toml
[server]
port = 8501  # Change to different port if needed
```

## Demo Flow

Perfect for showcasing the system:

1. **Start with empty system** → Show 0 documents indexed
2. **Upload a sample policy PDF** → Show processing feedback
3. **Check stats** → Show increased document count
4. **Ask a question** → Demonstrate chat with sources
5. **Upload another document** → Show multi-document capability
6. **Ask a cross-document question** → Show knowledge synthesis

## Production Notes

For production deployment:
- Use a production WSGI server for FastAPI (e.g., Gunicorn)
- Configure authentication for the UI
- Set up HTTPS/SSL
- Use environment-specific configurations
- Enable logging and monitoring

## Support

If you encounter issues:
1. Check the API logs in Terminal 1
2. Check the Streamlit logs in Terminal 2
3. Verify your API keys are valid
4. Ensure all dependencies are installed

## Architecture

```
User Browser
    ↓
Streamlit UI (port 8501)
    ↓
FastAPI Server (port 8000)
    ↓
├─ Vector Store (ChromaDB)
├─ Knowledge Graph (NetworkX)
└─ LLM Provider (Gemini/OpenAI)
```

The UI communicates with the FastAPI backend via REST API calls, making it easy to swap or scale components independently.
