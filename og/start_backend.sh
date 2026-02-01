#!/bin/bash

# RAG Chatbot Backend Startup Script

echo "======================================"
echo "   RAG Chatbot - Starting Backend"
echo "======================================"
echo ""

# Create input folder if it doesn't exist
if [ ! -d "inputfiles" ]; then
    echo "📁 Creating inputfiles directory..."
    mkdir -p inputfiles
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    echo "Please install Python 3.8+"
    exit 1
fi

# Download NLTK data if needed
echo "📚 Checking NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null

echo ""
echo "✅ Starting FastAPI Backend..."
echo "📊 API Docs: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start FastAPI server
python3 -m app.main
