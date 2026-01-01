# 📮 Postman Quick Reference Card

## 🔗 Base URL
```
http://localhost:8000
```

---

## 📍 Endpoints Summary

| # | Method | Endpoint | Purpose |
|---|--------|----------|---------|
| 1 | GET | `/api/v1/health` | Check server status |
| 2 | GET | `/api/v1/` | API info |
| 3 | POST | `/api/v1/documents/upload` | Upload PDF |
| 4 | POST | `/api/v1/chat` | Ask questions |
| 5 | GET | `/api/v1/stats` | System stats |

---

## 1. Health Check ✅

**GET** `http://localhost:8000/api/v1/health`

- Headers: None
- Body: None

---

## 2. Upload Document 📄

**POST** `http://localhost:8000/api/v1/documents/upload`

**Headers:**
```
Content-Type: application/json
```

**Body (raw JSON):**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_leave_policy.pdf"
}
```

**⚠️ Important**: Use absolute path (full path from root)

---

## 3. Chat Query 💬

**POST** `http://localhost:8000/api/v1/chat`

**Headers:**
```
Content-Type: application/json
```

**Body (raw JSON):**
```json
{
  "query": "What is the maternity leave policy?"
}
```

---

## 🧪 Quick Test Queries

### Copy & Paste These:

**Query 1: Maternity Leave**
```json
{"query": "How many weeks of maternity leave are provided and what percentage is paid?"}
```

**Query 2: 401k Benefits**
```json
{"query": "What is the company 401k match percentage and vesting schedule?"}
```

**Query 3: Remote Work**
```json
{"query": "What are the eligibility requirements for remote work?"}
```

**Query 4: Vacation Days**
```json
{"query": "How many vacation days do I get after 5 years of service?"}
```

**Query 5: Performance Review**
```json
{"query": "When is the annual performance review and what merit increase can I expect for exceeding expectations?"}
```

**Query 6: Health Insurance**
```json
{"query": "What health insurance plans are available and how much do they cost?"}
```

---

## 📂 Document Upload Paths

Update these with your actual paths:

**Leave Policy:**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_leave_policy.pdf"
}
```

**Benefits Policy:**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_benefits_policy.pdf"
}
```

**Remote Work Policy:**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/remote_work_policy.pdf"
}
```

**Performance Review Policy:**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/performance_review_policy.pdf"
}
```

---

## 🔄 Testing Sequence

1. ✅ **Health Check** → GET `/api/v1/health`
2. 📄 **Upload Doc 1** → POST `/api/v1/documents/upload` (Leave Policy)
3. 📄 **Upload Doc 2** → POST `/api/v1/documents/upload` (Benefits Policy)
4. 📄 **Upload Doc 3** → POST `/api/v1/documents/upload` (Remote Work)
5. 📄 **Upload Doc 4** → POST `/api/v1/documents/upload` (Performance)
6. 💬 **Ask Question** → POST `/api/v1/chat`
7. 📊 **Check Stats** → GET `/api/v1/stats`

---

## ⏱️ Expected Response Times

- Health Check: < 1 second
- Upload Document: 30-60 seconds ⏳
- Chat Query: 3-8 seconds
- Stats: < 1 second

---

## 🚨 Common Issues

**❌ "Connection refused"**
→ Start server: `python -m app.main`

**❌ "File not found"**
→ Use absolute path (starts with `/`)

**❌ Timeout**
→ Increase Postman timeout to 120 seconds

---

## ✨ Success Indicators

**Upload Response:**
```json
{
  "status": "processed",
  "chunks_created": 15,  ← Should be > 10
  "entities_extracted": 8  ← Should be > 5
}
```

**Chat Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "document_name": "employee_leave_policy.pdf",
      "relevance_score": 0.94  ← Should be > 0.7
    }
  ]
}
```

---

**📖 Full Guide**: See `POSTMAN_GUIDE.md` for complete details
