#!/bin/bash

# RAG Chatbot Frontend Startup Script

echo "======================================"
echo "   RAG Chatbot - Starting Frontend"
echo "======================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Check if backend is running
echo "🔍 Checking backend connection..."
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend not detected at http://localhost:8000"
    echo "Please start the backend first:"
    echo "  ./start_backend.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ Starting Streamlit UI..."
echo "🌐 UI will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run ui_api.py

# or use python 3 
python3 -m streamlit run ui.py --server.headless true
