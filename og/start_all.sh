#!/bin/bash

# RAG Chatbot - Start Both Backend and Frontend

echo "================================================"
echo "   RAG Chatbot - Starting Full Application"
echo "================================================"
echo ""

# Check if tmux or screen is available for multi-process management
if command -v tmux &> /dev/null; then
    echo "Using tmux for process management..."
    echo ""

    # Create new tmux session with backend
    tmux new-session -d -s rag_chatbot -n backend "./start_backend.sh"

    # Split window and start frontend
    tmux split-window -h -t rag_chatbot "sleep 5 && ./start_frontend.sh"

    echo "✅ Both services started in tmux session 'rag_chatbot'"
    echo ""
    echo "📊 Backend API: http://localhost:8000/docs"
    echo "🌐 Frontend UI: http://localhost:8501"
    echo ""
    echo "To attach to the session: tmux attach -t rag_chatbot"
    echo "To stop all services: tmux kill-session -t rag_chatbot"
    echo ""

    # Attach to the session
    tmux attach -t rag_chatbot

else
    echo "⚠️  tmux not found. Please install tmux for better process management."
    echo ""
    echo "Manual startup:"
    echo ""
    echo "Terminal 1:"
    echo "  ./start_backend.sh"
    echo ""
    echo "Terminal 2 (after backend starts):"
    echo "  ./start_frontend.sh"
    echo ""
fi
