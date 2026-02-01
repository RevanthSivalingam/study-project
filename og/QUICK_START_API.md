# 🚀 Quick Start - API Version

Get your RAG chatbot running with backend + frontend architecture!

---

## ⚡ Super Quick Start (1 Command)

```bash
# Install dependencies first time only
pip install -r requirements.txt

# Start everything
./start_all.sh
```

**Requires:** `tmux` (install with `brew install tmux`)

---

## 📝 Step-by-Step Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"
```

### Step 2: Start Backend Server

**Terminal 1:**
```bash
./start_backend.sh
```

**Wait for:**
```
✅ Starting FastAPI Backend...
📊 API Docs: http://localhost:8000/docs
```

### Step 3: Start Frontend UI

**Terminal 2:**
```bash
./start_frontend.sh
```

**Wait for:**
```
✅ Starting Streamlit UI...
🌐 UI will open at: http://localhost:8501
```

### Step 4: Use the Application

1. **Open:** http://localhost:8501
2. **Upload documents** via sidebar
3. **Click:** "Upload & Process"
4. **Ask questions** in the chat

---

## 🎯 What You Get

### Backend (Port 8000)
- ✅ REST API endpoints
- ✅ FastAPI with auto docs
- ✅ Request validation
- ✅ CORS enabled
- ✅ Health checks

### Frontend (Port 8501)
- ✅ Chat interface
- ✅ Document upload
- ✅ Source citations
- ✅ Real-time stats
- ✅ Clean UI

---

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:8501 | Main chat interface |
| **Backend API** | http://localhost:8000 | API server |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/api/v1/health | Backend status |

---

## 🛑 Stopping Services

### If Using `start_all.sh` (tmux)
```bash
# Kill the tmux session
tmux kill-session -t rag_chatbot
```

### If Using Separate Terminals
```bash
# Press Ctrl+C in each terminal window
# Or kill processes by port:
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:8501 | xargs kill -9  # Frontend
```

---

## 🔍 Verify Everything Works

### 1. Check Backend
```bash
curl http://localhost:8000/api/v1/health
# Should return: {"status":"healthy","initialized":false}
```

### 2. Check Frontend
- Open http://localhost:8501
- Should see "✅ API Connected" in sidebar

### 3. Upload Document
- Click "Choose PDF/TXT/JSON files"
- Select a file
- Click "Upload & Process"
- Wait for success message

### 4. Test Query
- Type: "What is in the document?"
- Press Enter
- See answer with sources

---

## 🆚 Two Versions Available

### Standalone Version (`ui.py`)
```bash
./start_ui.sh
```
- ✅ Simple: One process
- ✅ Fast: No API overhead
- ❌ No API: Can't integrate with other apps

### API Version (`ui_api.py` + backend)
```bash
./start_all.sh
```
- ✅ Scalable: Separate services
- ✅ API Available: Integrate with anything
- ✅ Production Ready: Better architecture
- ❌ Complex: Two processes needed

---

## 🐛 Troubleshooting

### "API Offline" Error

**Problem:** Frontend can't connect to backend

**Solution:**
1. Check if backend is running: `curl http://localhost:8000/api/v1/health`
2. If not, start it: `./start_backend.sh`
3. Refresh frontend page

---

### Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

---

### Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

### tmux Not Found

**Problem:** `command not found: tmux`

**Solution:**
```bash
# Install tmux
brew install tmux

# Or use manual startup (2 terminals)
./start_backend.sh    # Terminal 1
./start_frontend.sh   # Terminal 2
```

---

## 💡 Pro Tips

### Background Mode
```bash
# Start backend in background
nohup ./start_backend.sh > backend.log 2>&1 &

# Start frontend in background
nohup ./start_frontend.sh > frontend.log 2>&1 &
```

### Check Logs
```bash
# View backend log
tail -f backend.log

# View frontend log
tail -f frontend.log
```

### Quick Restart
```bash
# Kill all
tmux kill-session -t rag_chatbot

# Restart
./start_all.sh
```

---

## 📚 Next Steps

1. **Read:** API_ARCHITECTURE.md for detailed architecture info
2. **Test:** API at http://localhost:8000/docs
3. **Explore:** Try different queries
4. **Customize:** Modify endpoints in `app/main.py`

---

## ✅ Checklist

First time setup:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] tmux installed (optional)

Every time:
- [ ] Backend started (port 8000)
- [ ] Frontend started (port 8501)
- [ ] Documents uploaded
- [ ] System initialized

---

**Architecture:** Backend + Frontend
**Startup Time:** ~10 seconds
**Ready to use!** 🎉
