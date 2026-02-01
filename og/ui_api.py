"""
Streamlit UI for RAG Chatbot (API Version)
Connects to FastAPI backend via REST API
"""

import streamlit as st
import requests
import os
from pathlib import Path
from typing import Dict, Any
import shutil

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page Configuration
st.set_page_config(
    page_title="RAG Policy Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .method-kg {
        color: #2E7D32;
        font-weight: bold;
    }
    .method-semantic {
        color: #1976D2;
        font-weight: bold;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .stats-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #F57C00;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #C62828;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# API Functions
def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_stats() -> Dict[str, Any]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json().get("data", {})
    except:
        pass
    return {}

def upload_document(file_path: str, document_type: str, metadata: Dict = None) -> Dict:
    """Upload document to API"""
    payload = {
        "file_path": file_path,
        "document_type": document_type,
        "metadata": metadata or {}
    }
    response = requests.post(f"{API_BASE_URL}/documents/upload", json=payload)
    return response.json()

def send_chat_message(query: str) -> Dict:
    """Send chat message to API"""
    payload = {
        "query": query,
        "session_id": st.session_state.session_id
    }
    response = requests.post(f"{API_BASE_URL}/chat", json=payload)
    return response.json()

def initialize_system(api_key: str = None) -> Dict:
    """Initialize the RAG system"""
    payload = {
        "api_key": api_key,
        "input_folder": "inputfiles"
    }
    response = requests.post(f"{API_BASE_URL}/initialize", json=payload)
    return response.json()

def format_confidence(score: float) -> str:
    """Format confidence score with color"""
    if score >= 0.8:
        return f'<span class="confidence-high">{score:.2%}</span>'
    elif score >= 0.6:
        return f'<span class="confidence-medium">{score:.2%}</span>'
    else:
        return f'<span class="confidence-low">{score:.2%}</span>'

def format_method(method: str) -> str:
    """Format retrieval method with color"""
    if method == "knowledge_graph":
        return '<span class="method-kg">📊 Knowledge Graph</span>'
    elif method == "semantic_search":
        return '<span class="method-semantic">🔍 Semantic Search</span>'
    else:
        return method

# Main UI Layout
st.markdown('<div class="main-header">📚 RAG Policy Chatbot</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    # API Health Check
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.markdown("""
        <div class="error-box">
            <strong>❌ API Offline</strong><br>
            Please start the backend server:<br>
            <code>python -m app.main</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # OpenAI API Key (optional)
    st.markdown("### OpenAI API Key (Optional)")
    api_key = st.text_input(
        "Enter OpenAI API Key",
        type="password",
        help="Optional: For LLM-based answer generation"
    )

    st.markdown("---")

    # Document Upload Section
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF/TXT/JSON files",
        type=["pdf", "txt", "json"],
        accept_multiple_files=True,
        help="Upload policy documents for the chatbot to learn from"
    )

    document_type = st.selectbox(
        "Document Type",
        ["policy", "procedure", "guideline", "manual", "other"]
    )

    category = st.text_input("Category (optional)", placeholder="e.g., HR, IT, Finance")
    department = st.text_input("Department (optional)", placeholder="e.g., Human Resources")

    if st.button("📤 Upload & Process", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            # Create input folder if it doesn't exist
            input_dir = Path("inputfiles")
            input_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = input_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(str(file_path))

            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

            # Process first file (triggers system initialization)
            with st.spinner("Processing documents..."):
                try:
                    metadata = {}
                    if category:
                        metadata["category"] = category
                    if department:
                        metadata["department"] = department

                    result = upload_document(saved_files[0], document_type, metadata)

                    if result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        st.info(f"📊 Sections created: {result.get('chunks_created', 0)}")
                        st.info(f"🔍 Entities extracted: {result.get('entities_extracted', 0)}")
                    else:
                        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # Initialize System Button
    if st.button("🔄 Initialize System", type="secondary"):
        with st.spinner("Initializing..."):
            try:
                result = initialize_system(api_key if api_key else None)

                if result.get("success"):
                    st.success(result.get("message"))
                    if result.get("stats"):
                        stats = result["stats"]
                        st.info(f"📄 Documents: {stats.get('documents', 0)}")
                        st.info(f"📑 Sections: {stats.get('sections', 0)}")
                else:
                    st.error(result.get("message"))
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # System Statistics
    st.header("📊 System Stats")
    stats = get_stats()

    if stats and stats.get("total_documents_indexed", 0) > 0:
        st.markdown(f"""
        <div class="stats-box">
            <strong>Documents Indexed:</strong> {stats.get('total_documents_indexed', 0)}<br>
            <strong>Sections Extracted:</strong> {stats.get('total_sections', 0)}<br>
            <strong>Clusters Created:</strong> {stats.get('total_clusters', 0)}<br>
            <strong>Key Terms Learned:</strong> {stats.get('learned_terms', 0)}<br>
            <strong>Knowledge Graph Entities:</strong> {stats.get('total_entities', 0)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📋 No documents loaded yet")

    st.markdown("---")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Clear All Data Button
    if st.button("⚠️ Clear All Data", type="secondary"):
        if os.path.exists("inputfiles"):
            shutil.rmtree("inputfiles")
        st.session_state.messages = []
        st.success("All data cleared!")
        st.rerun()

# Main Chat Area
st.header("💬 Chat with Your Documents")

# Show initialization status
stats = get_stats()
if not stats or stats.get("total_documents_indexed", 0) == 0:
    st.markdown("""
    <div class="info-box">
        <strong>👋 Getting Started:</strong><br>
        1. Ensure the backend server is running (<code>python -m app.main</code>)<br>
        2. Upload documents using the sidebar (PDF, TXT, or JSON)<br>
        3. Click "Upload & Process" to initialize the system<br>
        4. Start asking questions about your documents!
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]

            # Show retrieval method
            if metadata.get("method"):
                st.markdown(
                    f"**Method:** {format_method(metadata['method'])}",
                    unsafe_allow_html=True
                )

            # Show confidence
            if "confidence_score" in metadata:
                st.markdown(
                    f"**Confidence:** {format_confidence(metadata['confidence_score'])}",
                    unsafe_allow_html=True
                )

            # Show section title
            if metadata.get("section_title"):
                st.markdown(f"**Section:** {metadata['section_title']}")

            # Show sources
            if metadata.get("sources"):
                with st.expander("📎 View Sources"):
                    for idx, source in enumerate(metadata["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {idx}:</strong> {source.get('document_name', 'Unknown')}<br>
                            <strong>Relevance:</strong> {source.get('relevance_score', 0):.2%}<br>
                            <em>"{source.get('excerpt', '')[:150]}..."</em>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
stats = get_stats()
chat_disabled = not stats or stats.get("total_documents_indexed", 0) == 0

if prompt := st.chat_input("Ask a question about your documents...", disabled=chat_disabled):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = send_chat_message(prompt)

                if "detail" in response:
                    # Error response
                    error_msg = f"❌ Error: {response['detail']}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                else:
                    # Success response
                    answer = response.get("answer", "No answer received")
                    st.markdown(answer)

                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "method": response.get("method"),
                            "section_title": response.get("section_title"),
                            "sources": response.get("sources", []),
                            "confidence_score": response.get("confidence_score", 0)
                        }
                    })

                    # Display metadata
                    method = response.get("method")
                    if method:
                        st.markdown(
                            f"**Method:** {format_method(method)}",
                            unsafe_allow_html=True
                        )

                    confidence = response.get("confidence_score", 0)
                    st.markdown(
                        f"**Confidence:** {format_confidence(confidence)}",
                        unsafe_allow_html=True
                    )

                    section_title = response.get("section_title")
                    if section_title:
                        st.markdown(f"**Section:** {section_title}")

                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📎 View Sources"):
                            for idx, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {idx}:</strong> {source.get('document_name', 'Unknown')}<br>
                                    <strong>Relevance:</strong> {source.get('relevance_score', 0):.2%}<br>
                                    <em>"{source.get('excerpt', '')[:150]}..."</em>
                                </div>
                                """, unsafe_allow_html=True)

                    st.rerun()

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "RAG Policy Chatbot | Knowledge-Guided Retrieval | API-Powered Architecture"
    "</div>",
    unsafe_allow_html=True
)
