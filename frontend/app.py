import os
from typing import Dict, List, Optional
import streamlit as st

from api_client import RAGApiClient


# ------------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Express Analytics | RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom light-mode modern CSS with subtle teal accents
CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #FAFAFB;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* Brand Header in Sidebar */
    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.25rem;
    }

    /* Status Badges */
    .badge-online {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F766E;
    }

    /* Citations Chip */
    .citation-chip {
        display: inline-block;
        background-color: #F0FDFA;
        color: #0F766E;
        border: 1px solid #CCFBF1;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.78rem;
        font-family: monospace;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Fallback Warning Box */
    .fallback-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        color: #92400E;
        font-size: 0.88rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Execution Details Pill */
    .exec-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-family: monospace;
        margin-right: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------------------------------
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

if "api_client" not in st.session_state:
    st.session_state.api_client = RAGApiClient(base_url=st.session_state.api_url)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}

if "feedback_active_query" not in st.session_state:
    st.session_state.feedback_active_query = None

if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = None


# ------------------------------------------------------------------------------
# Sidebar Navigation & System Status
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">⚡ Express Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">RAG Technical Docs Assistant</div>', unsafe_allow_html=True)

    # Health check
    health_result = st.session_state.api_client.check_health()
    if health_result.get("healthy"):
        st.markdown(
            '<div class="badge-online">🟢 Backend Connected (FastAPI)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-offline">🔴 Backend Offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Server check error: {health_result.get('error', 'Unreachable')}")
        if st.button("🔄 Retry Connection", key="retry_health"):
            st.session_state.api_client = RAGApiClient(base_url=st.session_state.api_url)
            st.rerun()

    st.markdown("---")

    # Navigation Menu
    nav_selection = st.radio(
        "Navigation",
        options=["💬 Chat Assistant", "📁 Document Explorer", "📤 Upload Documents"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Session Stats & Clear Action
    st.markdown("### Workspace Tools")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_submitted = {}
        st.session_state.feedback_active_query = None
        st.rerun()

    with st.expander("⚙️ Backend Settings"):
        custom_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_url,
            help="FastAPI server endpoint (default: http://localhost:8000)",
        )
        if custom_url != st.session_state.api_url:
            st.session_state.api_url = custom_url
            st.session_state.api_client = RAGApiClient(base_url=custom_url)
            st.success("API Base URL updated!")
            st.rerun()

    st.caption("Powered by LangGraph, ChromaDB & Google Gemini")


# ------------------------------------------------------------------------------
# View 1: Chat Assistant
# ------------------------------------------------------------------------------
if nav_selection == "💬 Chat Assistant":
    st.title("Technical Documentation Assistant")
    st.caption(
        "Ask questions about technical documents in your corpus (FastAPI, Pydantic, Python Typing, LangGraph). "
        "The system uses self-corrective LangGraph workflows to retrieve, grade, and ground answers strictly in citations."
    )

    # Quick prompt suggestion chips if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📌 How do I validate models in Pydantic?", use_container_width=True):
                st.session_state.prefill_question = "How do I validate models with Pydantic BaseModel?"
                st.rerun()
            if st.button("📌 How does FastAPI handle Dependency Injection?", use_container_width=True):
                st.session_state.prefill_question = "How does FastAPI handle dependency injection with Depends?"
                st.rerun()
        with col2:
            if st.button("📌 What are StateGraph nodes and edges in LangGraph?", use_container_width=True):
                st.session_state.prefill_question = "What are the core concepts of StateGraph, nodes, and edges in LangGraph?"
                st.rerun()
            if st.button("🍪 Unrelated: What is a chocolate chip cookie recipe?", use_container_width=True):
                st.session_state.prefill_question = "What is the secret recipe for baking chocolate chip cookies?"
                st.rerun()
        st.markdown("---")

    # Render Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "⚡"):
            if role == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                # Assistant message rendering
                st.markdown(msg["content"])

                # Fallback Notice
                if msg.get("is_fallback"):
                    st.markdown(
                        """
                        <div class="fallback-box">
                            <strong>⚠️ Context Insufficient:</strong> The indexed technical corpus does not contain sufficient details to answer this query safely without hallucinating.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Source Citations
                citations = msg.get("citations", [])
                if citations:
                    st.markdown("**📚 Sources cited:**")
                    chips_html = "".join([f'<span class="citation-chip">📄 {c}</span>' for c in citations])
                    st.markdown(chips_html, unsafe_allow_html=True)

                # Execution Details Expander
                query_id = msg.get("query_id")
                with st.expander("🔍 Workflow Execution Details"):
                    st.markdown(f"**Query ID:** `{query_id or 'N/A'}`")
                    st.markdown(f"**Query Intent:** `{msg.get('query_intent', 'N/A')}`")
                    st.markdown(f"**Search Query Used:** `{msg.get('search_query', 'N/A')}`")
                    st.markdown(f"**Self-Correction Retries:** `{msg.get('retry_count', 0)} / 2`")
                    st.markdown(f"**Relevant Chunks Graded:** `{msg.get('relevant_chunks_count', 0)} / {msg.get('retrieved_chunks_count', 0)}`")
                    st.markdown(f"**Safe Fallback Triggered:** `{msg.get('is_fallback', False)}`")

                # Feedback Buttons
                if query_id:
                    feedback_key = f"fb_{query_id}"
                    submitted_record = st.session_state.feedback_submitted.get(query_id)

                    if submitted_record:
                        st.markdown(
                            f"<small style='color: #059669;'>✅ Feedback recorded: <strong>{submitted_record.get('rating')}</strong></small>",
                            unsafe_allow_html=True,
                        )
                    else:
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
                        with fb_col1:
                            if st.button("👍 Helpful", key=f"btn_up_{idx}"):
                                st.session_state.feedback_active_query = (query_id, "thumbs_up")
                        with fb_col2:
                            if st.button("👎 Inaccurate", key=f"btn_down_{idx}"):
                                st.session_state.feedback_active_query = (query_id, "thumbs_down")

                        # If this message was selected for feedback, show comment form
                        if (
                            st.session_state.feedback_active_query
                            and st.session_state.feedback_active_query[0] == query_id
                        ):
                            rating_choice = st.session_state.feedback_active_query[1]
                            with st.form(key=f"feedback_form_{query_id}"):
                                st.caption(f"Submitting rating: **{rating_choice}**")
                                comment_val = st.text_input(
                                    "Optional comment:",
                                    placeholder="e.g. Accurate citation or missing detail...",
                                    key=f"comment_input_{query_id}",
                                )
                                submit_btn = st.form_submit_button("Send Feedback")
                                if submit_btn:
                                    fb_resp = st.session_state.api_client.submit_feedback(
                                        query_id=query_id,
                                        rating=rating_choice,
                                        comment=comment_val if comment_val.strip() else None,
                                    )
                                    if fb_resp.get("success"):
                                        st.session_state.feedback_submitted[query_id] = {
                                            "rating": rating_choice,
                                            "comment": comment_val,
                                        }
                                        st.session_state.feedback_active_query = None
                                        st.success("Thank you! Feedback recorded.")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to submit feedback: {fb_resp.get('error')}")

    # Handle prefilled questions
    default_prompt = ""
    if st.session_state.prefill_question:
        default_prompt = st.session_state.prefill_question
        st.session_state.prefill_question = None

    # Chat Input
    user_input = st.chat_input("Ask a technical question about the documentation...", key="chat_input")
    active_prompt = user_input or default_prompt

    if active_prompt:
        # Add user query to conversation history
        st.session_state.messages.append({"role": "user", "content": active_prompt})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{active_prompt}**")

        # Call FastAPI backend with LangGraph workflow
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Analyzing question, retrieving from ChromaDB, and grading relevance..."):
                resp = st.session_state.api_client.query_assistant(question=active_prompt, max_retries=2)

            if resp.get("success"):
                data = resp["data"]
                answer_text = data.get("answer", "")
                st.markdown(answer_text)

                # Append assistant response to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "query_id": data.get("query_id"),
                        "citations": data.get("citations", []),
                        "is_fallback": data.get("is_fallback", False),
                        "retry_count": data.get("retry_count", 0),
                        "query_intent": data.get("query_intent", "conceptual"),
                        "search_query": data.get("search_query", active_prompt),
                        "retrieved_chunks_count": data.get("retrieved_chunks_count", 0),
                        "relevant_chunks_count": data.get("relevant_chunks_count", 0),
                    }
                )
                st.rerun()
            else:
                error_msg = f"❌ **Error executing query:** {resp.get('error')}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ------------------------------------------------------------------------------
# View 2: Document Explorer
# ------------------------------------------------------------------------------
elif nav_selection == "📁 Document Explorer":
    st.title("Document Corpus & Knowledge Store")
    st.caption("Inspect documents indexed in the ChromaDB persistent vector database.")

    # Refresh Button
    col_ref1, col_ref2 = st.columns([8, 2])
    with col_ref2:
        if st.button("🔄 Refresh Catalog", use_container_width=True):
            st.rerun()

    # Fetch document list from GET /api/v1/documents
    doc_resp = st.session_state.api_client.list_documents()

    if doc_resp.get("success"):
        data = doc_resp["data"]
        total_docs = data.get("total_documents", 0)
        total_chunks = data.get("total_chunks", 0)
        doc_list = data.get("documents", [])

        # Metric Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Documents</div>
                    <div class="metric-value">{total_docs}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Vector Chunks</div>
                    <div class="metric-value">{total_chunks}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Vector Database</div>
                    <div class="metric-value" style="font-size: 1.25rem;">ChromaDB (Cosine)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Indexed Files")

        # Filter box
        search_filter = st.text_input("🔍 Search document catalog by filename or title:", "")

        filtered_docs = [
            d for d in doc_list
            if search_filter.lower() in d.get("source_filename", "").lower()
            or search_filter.lower() in d.get("document_title", "").lower()
        ]

        if not filtered_docs:
            st.info("No documents found matching your filter.")
        else:
            for d in filtered_docs:
                filename = d.get("source_filename", "unnamed")
                title = d.get("document_title", "Untitled Document")
                file_type = d.get("file_type", ".txt")
                chunk_cnt = d.get("chunk_count", 0)
                char_cnt = d.get("total_characters", 0)
                source_path = d.get("source_path", "")

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 0.5rem; padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 600; font-size: 1.05rem; color: #111827;">{title}</span>
                                    <span class="citation-chip" style="margin-left: 0.5rem;">{file_type}</span>
                                </div>
                                <span style="font-size: 0.82rem; color: #6B7280;">📁 <code>{source_path}</code></span>
                            </div>
                            <div style="margin-top: 0.5rem; display: flex; gap: 1.5rem; font-size: 0.85rem; color: #4B5563;">
                                <span>📄 Filename: <code>{filename}</code></span>
                                <span>🧩 Chunks: <strong>{chunk_cnt}</strong></span>
                                <span>📝 Characters: <strong>{char_cnt:,}</strong></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.error(f"Failed to load documents: {doc_resp.get('error')}")


# ------------------------------------------------------------------------------
# View 3: Upload Documents
# ------------------------------------------------------------------------------
elif nav_selection == "📤 Upload Documents":
    st.title("Upload & Ingest Documentation")
    st.caption(
        "Upload new Markdown (.md), Plain Text (.txt), or HTML (.html) technical files. "
        "The system will automatically extract document titles, split them into semantic chunks with 150-char overlap, "
        "generate vector embeddings, and store them in ChromaDB with duplicate protection."
    )

    uploaded_files = st.file_uploader(
        "Choose technical documentation files",
        type=["md", "txt", "html"],
        accept_multiple_files=True,
        help="Upload Markdown (.md), Text (.txt), or HTML (.html) files.",
    )

    if uploaded_files:
        st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
        for f in uploaded_files:
            st.markdown(f"- 📄 `{f.name}` ({len(f.getvalue()):,} bytes)")

        if st.button("🚀 Process & Ingest Files", type="primary"):
            # Prepare files for multipart upload
            prepared = []
            for f in uploaded_files:
                file_bytes = f.getvalue()
                # Guess content type
                ext = os.path.splitext(f.name)[1].lower()
                ctype = "text/markdown" if ext == ".md" else "text/html" if ext == ".html" else "text/plain"
                prepared.append((f.name, file_bytes, ctype))

            with st.spinner("Ingesting, chunking, and indexing into ChromaDB..."):
                ingest_resp = st.session_state.api_client.upload_documents(prepared)

            if ingest_resp.get("success"):
                res_data = ingest_resp["data"]
                st.success("✅ Ingestion successfully completed!")

                col_u1, col_u2, col_u3, col_u4 = st.columns(4)
                with col_u1:
                    st.metric("Files Uploaded", len(res_data.get("uploaded_files", [])))
                with col_u2:
                    st.metric("Chunks Indexed", res_data.get("total_chunks_indexed", 0))
                with col_u3:
                    st.metric("Chunks Updated", res_data.get("total_chunks_updated", 0))
                with col_u4:
                    st.metric("Skipped Duplicates", res_data.get("total_chunks_skipped", 0))

                st.info(f"Total chunks in vector store: **{res_data.get('total_store_chunks', 0)}**")

                failed_files = res_data.get("failed_files", [])
                if failed_files:
                    st.warning("⚠️ Some files were skipped or rejected:")
                    for failed in failed_files:
                        st.markdown(f"- ❌ {failed}")
            else:
                st.error(f"❌ Ingestion failed: {ingest_resp.get('error')}")
