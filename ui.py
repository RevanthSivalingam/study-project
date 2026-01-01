"""
Streamlit UI for Enterprise Policy Chatbot
"""
import streamlit as st
import requests
import os
from pathlib import Path
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page Configuration
st.set_page_config(
    page_title="Enterprise Policy Chatbot",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

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

def format_confidence(score: float) -> str:
    """Format confidence score with color"""
    if score >= 0.8:
        return f'<span class="confidence-high">{score:.2%}</span>'
    elif score >= 0.6:
        return f'<span class="confidence-medium">{score:.2%}</span>'
    else:
        return f'<span class="confidence-low">{score:.2%}</span>'

# Main UI Layout
st.markdown('<div class="main-header">📚 Enterprise Policy Chatbot</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    # API Health Check
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline - Please start the server")
        st.code("python -m app.main", language="bash")
        st.stop()

    # Provider Info
    from config.settings import settings
    provider = settings.llm_provider.upper()
    st.info(f"🤖 Provider: **{provider}**")

    st.markdown("---")

    # Document Upload Section
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload policy documents for the chatbot to learn from"
    )

    document_type = st.selectbox(
        "Document Type",
        ["policy", "procedure", "guideline", "manual", "other"]
    )

    category = st.text_input("Category (optional)", placeholder="e.g., HR, IT, Finance")
    department = st.text_input("Department (optional)", placeholder="e.g., Human Resources")

    if st.button("📤 Upload & Process", type="primary", disabled=uploaded_file is None):
        if uploaded_file is not None:
            # Save uploaded file temporarily
            upload_dir = Path("data/pdfs")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document..."):
                try:
                    metadata = {}
                    if category:
                        metadata["category"] = category
                    if department:
                        metadata["department"] = department

                    result = upload_document(str(file_path), document_type, metadata)

                    if "document_id" in result:
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"📊 Chunks created: {result.get('chunks_created', 0)}")
                        st.info(f"🔍 Entities extracted: {result.get('entities_extracted', 0)}")
                    else:
                        st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # System Statistics
    st.header("📊 System Stats")
    stats = get_stats()

    if stats:
        st.markdown(f"""
        <div class="stats-box">
            <strong>Documents Indexed:</strong> {stats.get('total_documents_indexed', 0)}<br>
            <strong>Entities Extracted:</strong> {stats.get('total_entities', 0)}<br>
            <strong>Knowledge Base:</strong> Active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No statistics available")

    st.markdown("---")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Area
st.header("💬 Chat with Policies")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📎 View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {idx}:</strong> {source.get('document_name', 'Unknown')}<br>
                            <strong>Page:</strong> {source.get('page_number', 'N/A')}<br>
                            <strong>Relevance:</strong> {source.get('relevance_score', 0):.2%}<br>
                            <em>"{source.get('excerpt', '')[:150]}..."</em>
                        </div>
                        """, unsafe_allow_html=True)

            # Display confidence and entities
            if "confidence_score" in message:
                st.markdown(
                    f"**Confidence:** {format_confidence(message['confidence_score'])}",
                    unsafe_allow_html=True
                )

            if message.get("entities_found"):
                st.markdown(f"**Entities:** {', '.join(message['entities_found'])}")

# Chat input
if prompt := st.chat_input("Ask a question about your policies..."):
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
                        "sources": response.get("sources", []),
                        "confidence_score": response.get("confidence_score", 0),
                        "entities_found": response.get("entities_found", [])
                    })

                    # Display sources
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📎 View Sources"):
                            for idx, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {idx}:</strong> {source.get('document_name', 'Unknown')}<br>
                                    <strong>Page:</strong> {source.get('page_number', 'N/A')}<br>
                                    <strong>Relevance:</strong> {source.get('relevance_score', 0):.2%}<br>
                                    <em>"{source.get('excerpt', '')[:150]}..."</em>
                                </div>
                                """, unsafe_allow_html=True)

                    # Display confidence and entities
                    confidence = response.get("confidence_score", 0)
                    st.markdown(
                        f"**Confidence:** {format_confidence(confidence)}",
                        unsafe_allow_html=True
                    )

                    entities = response.get("entities_found", [])
                    if entities:
                        st.markdown(f"**Entities:** {', '.join(entities)}")

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
    "Enterprise Policy Chatbot | Powered by " + provider + " | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
