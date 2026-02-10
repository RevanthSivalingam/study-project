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

def initialize_rag_system(llm_config=None):
    """Initialize or reinitialize the RAG system"""
    with st.spinner("Initializing RAG system..."):
        rag_system = RAGSystem(input_folder="inputfiles", llm_config=llm_config)
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

    # LLM Configuration
    st.markdown("### LLM Configuration (Optional)")
    provider = st.selectbox(
        "LLM Provider",
        ["none", "openai", "gemini"],
        help="Select LLM provider for answer synthesis. 'none' uses retrieval-only mode."
    )

    api_key = None
    if provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password")
    elif provider == "gemini":
        api_key = st.text_input("Gemini API Key", type="password")

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
            llm_config = {
                'provider': provider,
                'openai_api_key': api_key if provider == 'openai' else None,
                'gemini_api_key': api_key if provider == 'gemini' else None
            }
            success, message = initialize_rag_system(llm_config)

            if success:
                st.success(message)
            else:
                st.error(message)

    st.markdown("---")

    # Initialize System Button
    if st.button("🔄 Initialize System", type="secondary"):
        llm_config = {
            'provider': provider,
            'openai_api_key': api_key if provider == 'openai' else None,
            'gemini_api_key': api_key if provider == 'gemini' else None
        }
        success, message = initialize_rag_system(llm_config)

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

            # Show sources with relevance scores
            sources = metadata.get("sources", [])
            if sources:
                # Calculate and display cumulative relevance
                cumulative_relevance = sum(s.get("relevance_score", 0.0) for s in sources) / len(sources)

                # Color code cumulative
                if cumulative_relevance >= 0.8:
                    cum_color = "🟢"
                    cum_level = "Very High"
                elif cumulative_relevance >= 0.65:
                    cum_color = "🟡"
                    cum_level = "High"
                elif cumulative_relevance >= 0.5:
                    cum_color = "🟠"
                    cum_level = "Medium"
                else:
                    cum_color = "🔴"
                    cum_level = "Low"

                st.markdown(f"**📊 Avg Relevance:** {cum_color} {cumulative_relevance:.1%} ({cum_level})")

                with st.expander(f"📎 View {len(sources)} Individual Sources"):
                    for source in sources:
                        relevance = source.get("relevance_score", 0.0)
                        rank = source.get("rank", 0)
                        text = source.get("text", "")

                        # Color code relevance
                        if relevance >= 0.8:
                            rel_badge = "🟢 Very High"
                        elif relevance >= 0.65:
                            rel_badge = "🟡 High"
                        elif relevance >= 0.5:
                            rel_badge = "🟠 Medium"
                        else:
                            rel_badge = "🔴 Low"

                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source #{rank}</strong> &nbsp;
                            <span style="color: #666; font-size: 0.9em;">Relevance: {rel_badge} ({relevance:.1%})</span><br>
                            <div style="margin-top: 4px; font-size: 0.95em;">
                                <em>"{text}"</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            elif metadata.get("retrieved_sentences"):
                # Fallback for old format without scores
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
                            "retrieved_sentences": result.get("retrieved_sentences", []),
                            "sources": result.get("sources", []),
                            "confidence": result.get("confidence", {}),
                            "llm_provider": result.get("llm_provider"),
                            "tokens_used": result.get("tokens_used", 0),
                            "fallback_used": result.get("fallback_used", False)
                        }
                    })

                    # Display metadata in a clean format
                    st.markdown("---")

                    # Confidence Score with color coding
                    confidence = result.get("confidence", {})
                    conf_score = confidence.get("score", 0.0)
                    conf_level = confidence.get("level", "Unknown")

                    if conf_score >= 0.85:
                        conf_color = "🟢"
                    elif conf_score >= 0.70:
                        conf_color = "🟡"
                    elif conf_score >= 0.55:
                        conf_color = "🟠"
                    else:
                        conf_color = "🔴"

                    # Calculate cumulative relevance from sources
                    sources = result.get("sources", [])
                    if sources:
                        cumulative_relevance = sum(s.get("relevance_score", 0.0) for s in sources) / len(sources)
                    else:
                        # Fallback to metadata
                        metadata_result = result.get("metadata", {})
                        cumulative_relevance = metadata_result.get("avg_sentence_relevance", 0.0)

                    # Color code cumulative relevance
                    if cumulative_relevance >= 0.8:
                        rel_color = "🟢"
                        rel_level = "Very High"
                    elif cumulative_relevance >= 0.65:
                        rel_color = "🟡"
                        rel_level = "High"
                    elif cumulative_relevance >= 0.5:
                        rel_color = "🟠"
                        rel_level = "Medium"
                    else:
                        rel_color = "🔴"
                        rel_level = "Low"

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**🎯 Confidence:** {conf_color} {conf_score:.1%} ({conf_level})")

                    with col2:
                        method = result.get("method")
                        st.markdown(f"**Method:** {format_method(method)}", unsafe_allow_html=True)

                    with col3:
                        llm_provider = result.get("llm_provider", "none")
                        st.markdown(f"**🤖 Provider:** {llm_provider.upper()}")

                    # Display cumulative relevance prominently
                    st.markdown(f"**📊 Avg Relevance:** {rel_color} {cumulative_relevance:.1%} ({rel_level}) · Based on {len(sources)} sources")

                    # Section info
                    section_title = result.get("section_title")
                    if section_title:
                        st.markdown(f"**📄 Source Section:** {section_title}")

                    # Tokens used
                    tokens_used = result.get("tokens_used", 0)
                    if tokens_used > 0:
                        st.markdown(f"**💰 Tokens Used:** {tokens_used}")

                    # Fallback warning
                    if result.get("fallback_used"):
                        st.info("⚠️ LLM unavailable, using retrieval-only mode")

                    # Display sources with relevance scores
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander(f"📎 View {len(sources)} Sources with Relevance Scores"):
                            for source in sources:
                                relevance = source.get("relevance_score", 0.0)
                                rank = source.get("rank", 0)
                                text = source.get("text", "")

                                # Color code relevance
                                if relevance >= 0.8:
                                    rel_badge = "🟢 Very High"
                                elif relevance >= 0.65:
                                    rel_badge = "🟡 High"
                                elif relevance >= 0.5:
                                    rel_badge = "🟠 Medium"
                                else:
                                    rel_badge = "🔴 Low"

                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source #{rank}</strong> &nbsp;
                                    <span style="color: #666; font-size: 0.9em;">Relevance: {rel_badge} ({relevance:.1%})</span><br>
                                    <div style="margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                                        <em>"{text}"</em>
                                    </div>
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
