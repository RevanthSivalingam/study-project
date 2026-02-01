"""
Streamlit UI for RAG Chatbot
Simple interface for document-based question answering
"""

import streamlit as st
import os
from pathlib import Path
from rag_logic import RAGSystem
import shutil

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def initialize_rag_system(api_key=None):
    """Initialize or reinitialize the RAG system"""
    with st.spinner("Initializing RAG system..."):
        rag_system = RAGSystem(api_key=api_key, input_folder="inputfiles")
        success, message = rag_system.initialize()

        if success:
            st.session_state.rag_system = rag_system
            st.session_state.initialized = True
            return True, message
        else:
            st.session_state.initialized = False
            return False, message

def get_system_stats():
    """Get system statistics"""
    if st.session_state.rag_system and st.session_state.initialized:
        rag = st.session_state.rag_system
        return {
            "documents": len(rag.documents),
            "sections": len(rag.sections),
            "clusters": rag.NUM_CLUSTERS,
            "learned_terms": len(rag.learned_terms),
            "kg_entities": len(rag.knowledge_graph)
        }
    return None

def format_method(method):
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

    # OpenAI API Key (optional)
    st.markdown("### OpenAI API Key (Optional)")
    api_key = st.text_input(
        "Enter OpenAI API Key",
        type="password",
        help="Optional: For LLM-based answer generation. Leave empty for retrieval-only mode."
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

    if st.button("📤 Upload & Process", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            # Create input folder if it doesn't exist
            input_dir = Path("inputfiles")
            input_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = input_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

            # Initialize/Reinitialize RAG system
            success, message = initialize_rag_system(api_key if api_key else None)

            if success:
                st.success(message)
            else:
                st.error(message)

    st.markdown("---")

    # Initialize System Button
    if st.button("🔄 Initialize System", type="secondary"):
        success, message = initialize_rag_system(api_key if api_key else None)

        if success:
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")

    # System Statistics
    st.header("📊 System Stats")
    stats = get_system_stats()

    if stats:
        st.markdown(f"""
        <div class="stats-box">
            <strong>Documents Loaded:</strong> {stats['documents']}<br>
            <strong>Sections Extracted:</strong> {stats['sections']}<br>
            <strong>Clusters Created:</strong> {stats['clusters']}<br>
            <strong>Key Terms Learned:</strong> {stats['learned_terms']}<br>
            <strong>Knowledge Graph Entities:</strong> {stats['kg_entities']}
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
        st.session_state.rag_system = None
        st.session_state.initialized = False
        st.session_state.messages = []
        st.success("All data cleared!")
        st.rerun()

# Main Chat Area
st.header("💬 Chat with Your Documents")

# Show initialization status
if not st.session_state.initialized:
    st.markdown("""
    <div class="info-box">
        <strong>👋 Getting Started:</strong><br>
        1. Upload documents using the sidebar (PDF, TXT, or JSON)<br>
        2. Click "Upload & Process" to initialize the system<br>
        3. Start asking questions about your documents!
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
            if "method" in metadata:
                st.markdown(
                    f"**Method:** {format_method(metadata['method'])}",
                    unsafe_allow_html=True
                )

            # Show section title
            if metadata.get("section_title"):
                st.markdown(f"**Section:** {metadata['section_title']}")

            # Show retrieved sentences
            if metadata.get("retrieved_sentences"):
                with st.expander("📎 View Retrieved Sentences"):
                    for idx, sentence in enumerate(metadata["retrieved_sentences"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Sentence {idx}:</strong><br>
                            <em>"{sentence}"</em>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.initialized):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.rag_system:
                    result = st.session_state.rag_system.answer_query(prompt)

                    answer = result.get("answer", "No answer generated")
                    st.markdown(answer)

                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "method": result.get("method"),
                            "section_title": result.get("section_title"),
                            "retrieved_sentences": result.get("retrieved_sentences", [])
                        }
                    })

                    # Display metadata
                    method = result.get("method")
                    st.markdown(
                        f"**Method:** {format_method(method)}",
                        unsafe_allow_html=True
                    )

                    section_title = result.get("section_title")
                    if section_title:
                        st.markdown(f"**Section:** {section_title}")

                    retrieved_sentences = result.get("retrieved_sentences", [])
                    if retrieved_sentences:
                        with st.expander("📎 View Retrieved Sentences"):
                            for idx, sentence in enumerate(retrieved_sentences, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Sentence {idx}:</strong><br>
                                    <em>"{sentence}"</em>
                                </div>
                                """, unsafe_allow_html=True)

                    st.rerun()
                else:
                    error_msg = "System not initialized. Please upload documents first."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

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
    "RAG Policy Chatbot | Knowledge-Guided Retrieval | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
