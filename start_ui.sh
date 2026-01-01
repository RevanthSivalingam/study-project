#!/bin/bash
# Quick start script for the UI

echo "🚀 Starting Enterprise Policy Chatbot UI..."
echo ""
echo "Prerequisites:"
echo "  1. API server must be running on port 8000"
echo "  2. Streamlit will start on port 8501"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if API is running
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ API server is running"
else
    echo "❌ API server is NOT running!"
    echo ""
    echo "Please start the API server first:"
    echo "  python -m app.main"
    echo ""
    exit 1
fi

echo ""
echo "🌐 Starting Streamlit UI..."
echo "📱 Open your browser to: http://localhost:8501"
echo ""

# Start Streamlit
streamlit run ui.py
