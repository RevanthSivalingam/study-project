#!/bin/bash

# Enterprise Policy Chatbot - Quick Start Script

echo "=================================="
echo "Enterprise Policy Chatbot Setup"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file with your OpenAI API key:"
    echo "   - OPENAI_API_KEY=your_key_here"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/pdfs data/chroma_db

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To start the server:"
echo "  python -m app.main"
echo ""
echo "Or with uvicorn:"
echo "  uvicorn app.main:app --reload"
echo ""
echo "Access the API:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo ""
echo "=================================="
