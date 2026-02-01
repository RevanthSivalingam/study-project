#!/bin/bash

# RAG Chatbot UI Startup Script

echo "======================================"
echo "   RAG Policy Chatbot - Starting UI"
echo "======================================"
echo ""

# Create input folder if it doesn't exist
if [ ! -d "inputfiles" ]; then
    echo "📁 Creating inputfiles directory..."
    mkdir -p inputfiles
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Download NLTK data if needed
echo "📚 Checking NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"

echo ""
echo "✅ Starting Streamlit UI..."
echo "🌐 UI will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run ui.py
