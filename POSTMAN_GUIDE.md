# 📮 Postman Testing Guide - Enterprise Policy Chatbot

Complete guide for testing the chatbot API using Postman

## 🌐 Base URL

```
http://localhost:8000
```

## 📋 API Endpoints

---

## 1️⃣ Health Check

**Check if the server is running and view system stats**

### Request Details
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/health`
- **Headers**: None required
- **Query Params**: None
- **Body**: None

### Response Example
```json
{
  "status": "healthy",
  "app_name": "Enterprise Policy Chatbot",
  "version": "1.0.0",
  "timestamp": "2024-01-01T10:30:00.123456",
  "services": {
    "vector_store": "operational",
    "knowledge_graph": "operational",
    "llm": "operational",
    "documents_indexed": "4",
    "entities_extracted": "35"
  }
}
```

---

## 2️⃣ Root Endpoint

**Get API information**

### Request Details
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/`
- **Headers**: None required
- **Query Params**: None
- **Body**: None

### Response Example
```json
{
  "message": "Enterprise Policy Chatbot API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

## 3️⃣ Upload Document

**Upload and process a policy document**

### Request Details
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/documents/upload`
- **Headers**:
  ```
  Content-Type: application/json
  ```
- **Query Params**: None
- **Body** (raw JSON):

```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_leave_policy.pdf",
  "document_type": "policy"
}
```

### Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | ✅ Yes | **Absolute path** to the PDF file |
| `document_type` | string | ❌ No | Type of document (default: "policy") |
| `metadata` | object | ❌ No | Additional metadata (optional) |

### Important Notes
- ⚠️ **Use ABSOLUTE paths** (not relative)
- ⚠️ File must exist at the specified location
- ⚠️ Takes 30-60 seconds per document (processing + AI extraction)

### Response Example (Success)
```json
{
  "document_id": "a1b2c3d4e5f6",
  "file_name": "employee_leave_policy.pdf",
  "status": "processed",
  "chunks_created": 15,
  "entities_extracted": 8,
  "message": "Document successfully processed and indexed"
}
```

### Response Example (Error)
```json
{
  "detail": "File not found: /path/to/file.pdf"
}
```

### Test All 4 Documents

**Request 1: Leave Policy**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_leave_policy.pdf"
}
```

**Request 2: Benefits Policy**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/employee_benefits_policy.pdf"
}
```

**Request 3: Remote Work Policy**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/remote_work_policy.pdf"
}
```

**Request 4: Performance Review Policy**
```json
{
  "file_path": "/Users/rsivalingam/workspace/study-project/data/pdfs/performance_review_policy.pdf"
}
```

---

## 4️⃣ Chat / Query Documents

**Ask questions and get answers with source references**

### Request Details
- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/chat`
- **Headers**:
  ```
  Content-Type: application/json
  ```
- **Query Params**: None
- **Body** (raw JSON):

```json
{
  "query": "What is the maternity leave policy?",
  "session_id": "user123"
}
```

### Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | ✅ Yes | The question to ask |
| `session_id` | string | ❌ No | Session ID for conversation tracking (optional) |

### Response Example
```json
{
  "answer": "The maternity leave policy at ACME Corporation provides 16 weeks (112 days) of leave. The first 12 weeks are paid at 100% of base salary, and weeks 13-16 are paid at 50% of base salary. Employees must have completed 6 months of continuous service to be eligible. The leave can be taken up to 4 weeks before the expected delivery date, with a minimum of 12 weeks required after delivery.",
  "sources": [
    {
      "document_name": "employee_leave_policy.pdf",
      "page_number": 4,
      "chunk_id": "a1b2c3d4e5f6_chunk_3",
      "relevance_score": 0.94,
      "excerpt": "4. MATERNITY LEAVE POLICY\n\n4.1 Eligibility:\nFemale employees who have completed 6 months of continuous service are eligible for maternity leave.\n\n4.2 Leave Duration:\n- Total maternity leave: 16 weeks (112 days)..."
    },
    {
      "document_name": "employee_leave_policy.pdf",
      "page_number": 4,
      "chunk_id": "a1b2c3d4e5f6_chunk_4",
      "relevance_score": 0.87,
      "excerpt": "4.3 Paid Leave:\n- First 12 weeks: 100% of base salary\n- Weeks 13-16: 50% of base salary\n- Benefits continue during entire leave period..."
    }
  ],
  "confidence_score": 0.91,
  "entities_found": [
    "maternity leave",
    "benefits",
    "HR"
  ]
}
```

---

## 5️⃣ System Statistics

**Get statistics about indexed documents and entities**

### Request Details
- **Method**: `GET`
- **URL**: `http://localhost:8000/api/v1/stats`
- **Headers**: None required
- **Query Params**: None
- **Body**: None

### Response Example
```json
{
  "status": "success",
  "data": {
    "total_documents_indexed": 4,
    "total_entities": 35
  }
}
```

---

## 🧪 Sample Test Queries

Copy these queries into Postman to test multi-document search:

### Test 1: Leave Policy Question
```json
{
  "query": "How many weeks of maternity leave are provided and what percentage is paid?"
}
```
**Expected Source**: `employee_leave_policy.pdf`

---

### Test 2: Benefits Question
```json
{
  "query": "What is the company 401k match percentage and vesting schedule?"
}
```
**Expected Source**: `employee_benefits_policy.pdf`

---

### Test 3: Multiple Benefits
```json
{
  "query": "What health insurance plans are available and how much do they cost?"
}
```
**Expected Source**: `employee_benefits_policy.pdf`

---

### Test 4: Remote Work Question
```json
{
  "query": "What are the eligibility requirements for remote work?"
}
```
**Expected Source**: `remote_work_policy.pdf`

---

### Test 5: Remote Work Equipment
```json
{
  "query": "What equipment does the company provide for remote workers and what is the home office stipend?"
}
```
**Expected Source**: `remote_work_policy.pdf`

---

### Test 6: Performance Review
```json
{
  "query": "When is the annual performance review conducted and how are ratings determined?"
}
```
**Expected Source**: `performance_review_policy.pdf`

---

### Test 7: Merit Increase
```json
{
  "query": "What merit increase can I expect if my performance rating is exceeds expectations?"
}
```
**Expected Source**: `performance_review_policy.pdf`

---

### Test 8: Paternity Leave
```json
{
  "query": "How do I apply for paternity leave and how many weeks are provided?"
}
```
**Expected Source**: `employee_leave_policy.pdf`

---

### Test 9: Vacation Days
```json
{
  "query": "How many vacation days do I get after 5 years of service?"
}
```
**Expected Source**: `employee_leave_policy.pdf`

---

### Test 10: Cross-Document Query
```json
{
  "query": "What are all the paid time off benefits including vacation, sick leave, and holidays?"
}
```
**Expected Sources**: Multiple documents (Leave Policy + Benefits Policy)

---

### Test 11: Sick Leave
```json
{
  "query": "How many sick days do I get per year and do I need a medical certificate?"
}
```
**Expected Source**: `employee_leave_policy.pdf`

---

### Test 12: Tuition Reimbursement
```json
{
  "query": "Does the company offer tuition reimbursement and how much?"
}
```
**Expected Source**: `employee_benefits_policy.pdf`

---

### Test 13: Remote Work Schedule
```json
{
  "query": "What are the core business hours for remote employees?"
}
```
**Expected Source**: `remote_work_policy.pdf`

---

### Test 14: Performance Improvement Plan
```json
{
  "query": "What happens if I receive a needs improvement rating?"
}
```
**Expected Source**: `performance_review_policy.pdf`

---

### Test 15: Complex Multi-Policy Query
```json
{
  "query": "What benefits do I get as a remote worker including equipment, reimbursements, and time off?"
}
```
**Expected Sources**: Multiple documents (Remote Work + Benefits + Leave)

---

## 📝 Postman Collection Setup

### Step-by-Step Collection Creation

**1. Create New Collection**
- Name: `Enterprise Policy Chatbot`
- Base URL Variable: `{{base_url}}` = `http://localhost:8000`

**2. Create Folders**
- System
- Document Management
- Chat Queries
- Test Scenarios

**3. Add Requests**

#### Folder: System
1. Health Check (GET)
2. Root Info (GET)
3. System Stats (GET)

#### Folder: Document Management
1. Upload Leave Policy (POST)
2. Upload Benefits Policy (POST)
3. Upload Remote Work Policy (POST)
4. Upload Performance Policy (POST)

#### Folder: Chat Queries
1. Maternity Leave Query (POST)
2. 401k Benefits Query (POST)
3. Remote Work Query (POST)
4. Performance Review Query (POST)

#### Folder: Test Scenarios
1. Multi-Document Search (POST)
2. Cross-Policy Query (POST)
3. Specific Detail Query (POST)

---

## 🔧 Environment Variables

### Create Postman Environment

**Name**: `Local Development`

| Variable | Initial Value | Current Value |
|----------|---------------|---------------|
| `base_url` | `http://localhost:8000` | `http://localhost:8000` |
| `api_version` | `v1` | `v1` |
| `leave_policy_path` | `/Users/rsivalingam/workspace/study-project/data/pdfs/employee_leave_policy.pdf` | (same) |
| `benefits_policy_path` | `/Users/rsivalingam/workspace/study-project/data/pdfs/employee_benefits_policy.pdf` | (same) |
| `remote_work_path` | `/Users/rsivalingam/workspace/study-project/data/pdfs/remote_work_policy.pdf` | (same) |
| `performance_path` | `/Users/rsivalingam/workspace/study-project/data/pdfs/performance_review_policy.pdf` | (same) |

### Using Variables in Requests

**URL Example:**
```
{{base_url}}/api/{{api_version}}/health
```

**Body Example:**
```json
{
  "file_path": "{{leave_policy_path}}"
}
```

---

## ✅ Testing Workflow

### Complete Test Sequence

1. **Check Server Health**
   - Request: `GET /api/v1/health`
   - Verify: `status: "healthy"`

2. **Upload Document 1: Leave Policy**
   - Request: `POST /api/v1/documents/upload`
   - Body: Leave policy file path
   - Verify: `chunks_created > 0`

3. **Upload Document 2: Benefits Policy**
   - Request: `POST /api/v1/documents/upload`
   - Body: Benefits policy file path
   - Verify: `entities_extracted > 0`

4. **Upload Document 3: Remote Work Policy**
   - Request: `POST /api/v1/documents/upload`
   - Body: Remote work policy file path

5. **Upload Document 4: Performance Policy**
   - Request: `POST /api/v1/documents/upload`
   - Body: Performance policy file path

6. **Check Stats**
   - Request: `GET /api/v1/stats`
   - Verify: `total_documents_indexed: 4`

7. **Test Query 1: Leave-Specific**
   - Request: `POST /api/v1/chat`
   - Body: Maternity leave question
   - Verify: Source is `employee_leave_policy.pdf`

8. **Test Query 2: Benefits-Specific**
   - Request: `POST /api/v1/chat`
   - Body: 401k question
   - Verify: Source is `employee_benefits_policy.pdf`

9. **Test Query 3: Multi-Document**
   - Request: `POST /api/v1/chat`
   - Body: Question spanning multiple policies
   - Verify: Multiple sources returned

---

## 🚨 Common Errors & Solutions

### Error 1: Connection Refused
```json
{
  "detail": "Connection refused"
}
```
**Solution**: Start the server first
```bash
python -m app.main
```

### Error 2: File Not Found
```json
{
  "detail": "File not found: /path/to/file.pdf"
}
```
**Solution**:
- Use absolute paths (not relative)
- Verify file exists: `ls -la /path/to/file.pdf`
- Update path in Postman request

### Error 3: Invalid API Key
```json
{
  "detail": "OpenAI API key not found"
}
```
**Solution**: Check `.env` file has valid `OPENAI_API_KEY`

### Error 4: Timeout
```
Request timeout after 30000ms
```
**Solution**:
- Increase Postman timeout (Settings → General → Request timeout)
- Document upload takes 30-60 seconds
- Set timeout to 120000ms (2 minutes)

---

## 📊 Expected Response Times

| Endpoint | Expected Time |
|----------|---------------|
| Health Check | < 100ms |
| Document Upload | 30-60 seconds |
| Chat Query | 3-8 seconds |
| Stats | < 100ms |

---

## 🔍 Response Validation

### What to Check in Responses

**Document Upload Response:**
- ✅ `status: "processed"`
- ✅ `chunks_created > 10`
- ✅ `entities_extracted > 5`

**Chat Response:**
- ✅ `answer` is not empty
- ✅ `sources` array has 1+ items
- ✅ `relevance_score > 0.7`
- ✅ `document_name` matches expected policy

**Health Check Response:**
- ✅ `status: "healthy"`
- ✅ All services: `"operational"`
- ✅ `documents_indexed` increases after uploads

---

## 💡 Pro Tips

1. **Save Responses**: Click "Save Response" in Postman to compare results

2. **Use Tests Tab**: Add automatic validation
   ```javascript
   pm.test("Status is healthy", function () {
       pm.response.to.have.status(200);
       pm.expect(pm.response.json().status).to.eql("healthy");
   });
   ```

3. **Use Pre-request Scripts**: Set dynamic variables
   ```javascript
   pm.environment.set("timestamp", new Date().toISOString());
   ```

4. **Collection Runner**: Run all tests sequentially
   - Click "Run Collection"
   - Select all requests
   - Set delay: 5000ms between requests

5. **Export Collection**: Share with team
   - Click "..." → Export
   - Choose Collection v2.1
   - Share JSON file

---

## 📥 Quick Import (Optional)

Want to skip manual setup? Create this JSON file and import into Postman:

**File**: `postman_collection.json`

```json
{
  "info": {
    "name": "Enterprise Policy Chatbot",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/api/v1/health"
      }
    },
    {
      "name": "Upload Document",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "url": "{{base_url}}/api/v1/documents/upload",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"file_path\": \"{{leave_policy_path}}\"\n}"
        }
      }
    },
    {
      "name": "Chat Query",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "url": "{{base_url}}/api/v1/chat",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"query\": \"What is the maternity leave policy?\"\n}"
        }
      }
    }
  ]
}
```

Import: Postman → Import → Upload File → Select JSON

---

**Happy Testing! 🚀**
