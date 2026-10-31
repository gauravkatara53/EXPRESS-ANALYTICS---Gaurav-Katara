import os
from typing import Dict, List, Optional
import streamlit as st

from api_client import RAGApiClient


# Page configuration
st.set_page_config(
    page_title="Express Analytics | RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme stylesheet
DARK_SAAS_CSS = """
<style>
    /* Google Fonts Inter & JetBrains Mono */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global App Base */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0F172A 0%, #080D19 65%, #04070E 100%) !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* ------------------------------------------------------------- */
    /* Streamlit Native Header & Top Bar Visibility Fix              */
    /* ------------------------------------------------------------- */
    /* Remove top rainbow decoration line */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Make the top Streamlit header completely transparent and unobtrusive */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        pointer-events: none !important;
        z-index: 10 !important;
    }
    /* Keep action buttons (deploy, menu) in the header clickable */
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    /* Streamlit Main Container Spacing - Clear the top header bar cleanly */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }

    /* Headings with Modern Typography */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.028em !important;
    }

    /* Gradient Brand Title */
    .brand-gradient-text {
        background: linear-gradient(135deg, #2DD4BF 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.22rem;
        letter-spacing: -0.025em;
    }

    /* ------------------------------------------------------------- */
    /* Sidebar Styling & Custom Elements                             */
    /* ------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #080C16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }

    /* Brand Header Box */
    .sidebar-brand-card {
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(22, 33, 56, 0.45) 0%, rgba(11, 17, 30, 0.5) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0.85rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .brand-icon-box {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.25) 0%, rgba(56, 189, 248, 0.12) 100%);
        border: 1px solid rgba(45, 212, 191, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px rgba(45, 212, 191, 0.25);
        font-size: 1.2rem;
    }
    .brand-version-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        background: rgba(45, 212, 191, 0.12);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.3);
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .brand-tagline {
        font-size: 0.72rem;
        color: #94A3B8;
        font-weight: 500;
        letter-spacing: 0.01em;
        margin-top: 0.15rem;
    }

    /* Sidebar Section Titles */
    .sidebar-section-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.15rem 0 0.5rem 0.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* System Telemetry Card */
    .telemetry-card {
        background: rgba(14, 21, 37, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0.75rem;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
    }
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.65rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.38rem;
        font-size: 0.76rem;
    }
    .telemetry-row:last-child {
        margin-bottom: 0;
    }
    .telemetry-key {
        color: #64748B;
        font-weight: 500;
    }
    .telemetry-val-healthy {
        color: #34D399;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .telemetry-val-highlight {
        color: #38BDF8;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
    }
    .telemetry-val {
        color: #CBD5E1;
        font-weight: 500;
    }

    /* Pulsing Status Dot */
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 8px #34D399;
        animation: pulse 2s infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Modern Navigation Tabs (CSS Radio Replacement) */
    div[data-testid="stRadio"] {
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stRadio"] > div {
        background-color: transparent !important;
        gap: 0.4rem !important;
    }
    /* Hide the default radio circle completely */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    /* Transform the label into a sleek full-width nav card */
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.7rem 0.95rem !important;
        border-radius: 0.6rem !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.01em !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(45, 212, 191, 0.08) !important;
        border-color: rgba(45, 212, 191, 0.3) !important;
        color: #F8FAFC !important;
        transform: translateX(3px) !important;
    }
    /* Active checked state */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(45, 212, 191, 0.16) 0%, rgba(14, 165, 233, 0.06) 100%) !important;
        border-color: rgba(45, 212, 191, 0.6) !important;
        border-left: 3.5px solid #2DD4BF !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35), inset 0 0 16px rgba(45, 212, 191, 0.1) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Miniature Corpus Document Rows in Sidebar */
    .mini-doc-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.65rem;
        background: rgba(18, 26, 44, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        margin-bottom: 0.3rem;
        font-size: 0.75rem;
        transition: all 0.15s ease;
    }
    .mini-doc-row:hover {
        background: rgba(24, 35, 58, 0.8);
        border-color: rgba(45, 212, 191, 0.3);
    }
    .doc-type-pill {
        display: inline-block;
        padding: 0.1rem 0.35rem;
        border-radius: 0.25rem;
        font-size: 0.65rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .doc-type-md {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .doc-type-html {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .doc-type-txt {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .chunk-count-badge {
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
    }

    /* Sidebar Footer Tags */
    .sidebar-footer {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }
    .tech-tag {
        display: inline-block;
        padding: 0.12rem 0.42rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0.3rem;
        font-size: 0.65rem;
        color: #64748B;
        font-weight: 500;
    }

    /* ------------------------------------------------------------- */
    /* Main Content Elements                                         */
    /* ------------------------------------------------------------- */

    /* Metric Cards */
    .metric-card-dark {
        background: linear-gradient(180deg, rgba(20, 29, 48, 0.8) 0%, rgba(13, 20, 36, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.85rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .metric-card-dark:hover {
        border-color: rgba(45, 212, 191, 0.4);
        transform: translateY(-2px);
    }
    .metric-label-dark {
        font-size: 0.74rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .metric-val-dark {
        font-size: 1.95rem;
        font-weight: 800;
        color: #2DD4BF;
        letter-spacing: -0.03em;
        font-feature-settings: "tnum";
    }

    /* Modern Document Card in Explorer */
    .doc-card-dark {
        background: rgba(16, 24, 40, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.85rem;
        padding: 1.35rem 1.6rem;
        margin-bottom: 0.95rem;
        transition: all 0.2s ease;
        backdrop-filter: blur(8px);
    }
    .doc-card-dark:hover {
        border-color: rgba(45, 212, 191, 0.5);
        background: rgba(20, 30, 52, 0.85);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5);
    }

    /* Citation Badges */
    .citation-chip-dark {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(45, 212, 191, 0.1);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.3);
        padding: 0.3rem 0.75rem;
        border-radius: 0.45rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.45rem;
        margin-top: 0.4rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.15s ease;
    }
    .citation-chip-dark:hover {
        background: rgba(45, 212, 191, 0.18);
        border-color: rgba(45, 212, 191, 0.6);
        transform: translateY(-1px);
    }

    /* Fallback Alert Box */
    .fallback-box-dark {
        background: rgba(245, 158, 11, 0.09);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #F59E0B;
        padding: 0.95rem 1.25rem;
        border-radius: 0.6rem;
        color: #FCD34D;
        font-size: 0.88rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    /* Buttons */
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #E2E8F0 !important;
        border-radius: 0.55rem !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton>button:hover {
        border-color: #2DD4BF !important;
        background-color: rgba(45, 212, 191, 0.1) !important;
        color: #2DD4BF !important;
        box-shadow: 0 0 16px rgba(45, 212, 191, 0.2) !important;
    }

    /* Primary Buttons */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #0D9488 0%, #0891B2 100%) !important;
        border: 1px solid rgba(45, 212, 191, 0.4) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35) !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #14B8A6 0%, #06B6D4 100%) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.85rem 0.5rem !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(15, 22, 38, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 0.6rem !important;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: rgba(14, 21, 37, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #F8FAFC !important;
        border-radius: 0.55rem !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 14px rgba(45, 212, 191, 0.3) !important;
    }

    /* Chat Input */
    [data-testid="stChatInput"] textarea {
        background-color: #0B101D !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #F8FAFC !important;
        border-radius: 0.85rem !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 18px rgba(45, 212, 191, 0.35) !important;
    }

    /* Monospace Code Styling */
    code {
        color: #38BDF8 !important;
        background-color: rgba(15, 23, 42, 0.75) !important;
        padding: 0.15rem 0.45rem !important;
        border-radius: 0.3rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.86em !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
"""
st.markdown(DARK_SAAS_CSS, unsafe_allow_html=True)


# Session state initialization
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


# Sidebar: telemetry and navigation
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div class="brand-icon-box">⚡</div>
                <div style="flex-grow: 1; min-width: 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span class="brand-gradient-text">Express RAG</span>
                        <span class="brand-version-pill">v1.0 PROD</span>
                    </div>
                    <div class="brand-tagline">AI Documentation Intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic backend health check & catalog stats
    health_result = st.session_state.api_client.check_health()
    is_healthy = bool(health_result.get("healthy"))

    # Fetch document catalog summary for telemetry
    sidebar_docs: List[Dict] = []
    total_docs_count = 0
    total_chunks_count = 0
    if is_healthy:
        doc_summary_resp = st.session_state.api_client.list_documents()
        if doc_summary_resp.get("success"):
            data_dict = doc_summary_resp.get("data", {})
            sidebar_docs = data_dict.get("documents", [])
            total_docs_count = data_dict.get("total_documents", len(sidebar_docs))
            total_chunks_count = data_dict.get("total_chunks", sum(d.get("chunk_count", 0) for d in sidebar_docs))

    # Telemetry Card
    if is_healthy:
        st.markdown(
            f"""
            <div class="telemetry-card">
                <div class="telemetry-header">
                    <span style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">System Telemetry</span>
                    <span style="display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.72rem; color: #34D399; font-weight: 600;">
                        <span class="pulse-dot"></span> Live
                    </span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">FastAPI Service</span>
                    <span class="telemetry-val-healthy">● Port 8000</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Inference LPU</span>
                    <span class="telemetry-val-highlight">Groq (Qwen 2.5)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Vector Store</span>
                    <span class="telemetry-val">ChromaDB ({total_chunks_count} chunks)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Workflow Engine</span>
                    <span class="telemetry-val" style="color: #A78BFA; font-weight: 600;">LangGraph Cyclic</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="telemetry-card" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(30, 15, 20, 0.7);">
                <div class="telemetry-header">
                    <span style="font-size: 0.68rem; font-weight: 700; color: #F87171; text-transform: uppercase;">Service Offline</span>
                    <span style="color: #F87171; font-size: 0.72rem; font-weight: 600;">● Disconnected</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.3rem;">FastAPI backend is unreachable on port 8000.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Reconnect Backend", key="retry_health", use_container_width=True):
            st.session_state.api_client = RAGApiClient(base_url=st.session_state.api_url)
            st.rerun()

    # Workspace Views Navigation
    st.markdown(
        """
        <div class="sidebar-section-title">
            <span>Workspace Views</span>
            <span style="font-size: 0.65rem; color: #475569; font-weight: 500;">3 views</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_selection = st.radio(
        "Workspace Navigation",
        options=["💬 Chat Assistant", "📁 Document Explorer", "📤 Upload Documents"],
        label_visibility="collapsed",
    )

    # Active Corpus Miniature Preview in Sidebar
    if is_healthy and sidebar_docs:
        st.markdown(
            f"""
            <div class="sidebar-section-title">
                <span>Active Knowledge Base</span>
                <span style="font-size: 0.65rem; color: #2DD4BF; font-weight: 600;">{total_docs_count} docs</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for d in sidebar_docs[:5]:
            fname = d.get("source_filename", "doc")
            ftype = d.get("file_type", ".txt").lower()
            badge_class = "doc-type-md" if ftype == ".md" else "doc-type-html" if ftype == ".html" else "doc-type-txt"
            badge_text = ftype.replace(".", "").upper()
            ccnt = d.get("chunk_count", 0)
            st.markdown(
                f"""
                <div class="mini-doc-row" title="{fname}">
                    <div style="display: flex; align-items: center; gap: 0.45rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 175px;">
                        <span class="doc-type-pill {badge_class}">{badge_text}</span>
                        <span style="color: #CBD5E1; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{fname}</span>
                    </div>
                    <span class="chunk-count-badge">{ccnt} ch</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if len(sidebar_docs) > 5:
            st.markdown(
                f"<div style='font-size: 0.7rem; color: #64748B; text-align: center; margin-top: 0.25rem;'>+ {len(sidebar_docs) - 5} more in Explorer</div>",
                unsafe_allow_html=True,
            )

    # Workspace Action Controls
    st.markdown(
        """
        <div class="sidebar-section-title">
            <span>Session Controls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_submitted = {}
        st.session_state.feedback_active_query = None
        st.rerun()

    with st.expander("⚙️ Network & Endpoint Settings"):
        custom_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_url,
            help="FastAPI server endpoint (default: http://localhost:8000)",
        )
        if custom_url != st.session_state.api_url:
            st.session_state.api_url = custom_url
            st.session_state.api_client = RAGApiClient(base_url=custom_url)
            st.success("Target updated!")
            st.rerun()

    # Sidebar Brand Footer
    st.markdown(
        """
        <div class="sidebar-footer">
            <div style="display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.45rem;">
                <span class="tech-tag">LangGraph</span>
                <span class="tech-tag">Groq LPU</span>
                <span class="tech-tag">ChromaDB</span>
                <span class="tech-tag">FastAPI</span>
            </div>
            <div style="font-size: 0.68rem; color: #475569; font-weight: 500;">
                Express Analytics AI/ML Take-Home
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# View 1: Chat Assistant
if nav_selection == "💬 Chat Assistant":
    st.markdown(
        """
        <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="display: inline-block; padding: 0.22rem 0.65rem; border-radius: 9999px; background: rgba(45, 212, 191, 0.12); color: #2DD4BF; border: 1px solid rgba(45, 212, 191, 0.3); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;">SELF-CORRECTIVE RAG</span>
                <span style="display: inline-block; padding: 0.22rem 0.65rem; border-radius: 9999px; background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;">ZERO-HALLUCINATION GUARD</span>
            </div>
            <h1 style="margin-bottom: 0.35rem; font-size: 2.15rem; letter-spacing: -0.025em;">Technical Documentation Assistant</h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0; line-height: 1.5; max-width: 900px;">
                Ask technical questions about your documentation corpus. Built with self-corrective LangGraph workflows:
                intent classification, semantic retrieval, relevance grading, iterative query rewriting, and citation grounding.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick prompt suggestion chips if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown(
            "<span style='font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;'>Suggested Inquiries & Benchmark Tests</span>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧩 [Pydantic V2] How do I validate models with BaseModel?", use_container_width=True):
                st.session_state.prefill_question = "How do I validate models with Pydantic BaseModel?"
                st.rerun()
            if st.button("⚡ [FastAPI] How does Dependency Injection work with Depends?", use_container_width=True):
                st.session_state.prefill_question = "How does FastAPI handle dependency injection with Depends?"
                st.rerun()
        with col2:
            if st.button("🔄 [LangGraph] What are StateGraph, nodes, and cyclical edges?", use_container_width=True):
                st.session_state.prefill_question = "What are the core concepts of StateGraph, nodes, and edges in LangGraph?"
                st.rerun()
            if st.button("🛡️ [Fallback Guard] Out-of-Scope: Recipe for chocolate chip cookies?", use_container_width=True):
                st.session_state.prefill_question = "What is the secret recipe for baking chocolate chip cookies?"
                st.rerun()
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Render Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "⚡"):
            if role == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                # Assistant message body
                st.markdown(msg["content"])

                # Fallback Notice (if triggered)
                if msg.get("is_fallback"):
                    st.markdown(
                        """
                        <div class="fallback-box-dark">
                            <strong>⚠️ Safe Non-Hallucination Fallback:</strong> The indexed technical corpus does not contain sufficient details to answer this query safely. The pipeline self-corrected and verified zero relevant context.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Source Citations
                citations = msg.get("citations", [])
                if citations:
                    st.markdown("<div style='margin-top: 0.6rem;'><strong style='color: #94A3B8; font-size: 0.85rem;'>📚 Source Documents Cited:</strong></div>", unsafe_allow_html=True)
                    chips_html = "".join([f'<span class="citation-chip-dark">📄 {c}</span>' for c in citations])
                    st.markdown(chips_html, unsafe_allow_html=True)

                # Execution Details Expander
                query_id = msg.get("query_id")
                with st.expander("🔍 Workflow Trace & Execution Details"):
                    st.markdown(
                        f"""
                        <div style="background: rgba(11, 17, 30, 0.7); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 0.6rem; padding: 0.85rem 1rem;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; font-size: 0.82rem;">
                                <div><span style="color: #64748B;">Query ID:</span> <code style="font-size: 0.75rem;">{query_id or 'N/A'}</code></div>
                                <div><span style="color: #64748B;">Query Intent:</span> <span style="color: #38BDF8; font-weight: 600;">{msg.get('query_intent', 'N/A')}</span></div>
                                <div><span style="color: #64748B;">Rewritten Search:</span> <code style="font-size: 0.75rem;">{msg.get('search_query', 'N/A')}</code></div>
                                <div><span style="color: #64748B;">Self-Correction Loops:</span> <span style="color: {'#34D399' if msg.get('retry_count', 0) == 0 else '#FCD34D'}; font-weight: 600;">{msg.get('retry_count', 0)} / 2</span></div>
                                <div><span style="color: #64748B;">Graded Relevance:</span> <span style="color: #2DD4BF; font-weight: 600;">{msg.get('relevant_chunks_count', 0)} / {msg.get('retrieved_chunks_count', 0)} chunks</span></div>
                                <div><span style="color: #64748B;">Fallback Guard:</span> <span style="color: {'#F87171' if msg.get('is_fallback') else '#34D399'}; font-weight: 600;">{'⚠️ Triggered' if msg.get('is_fallback') else '✅ Grounded'}</span></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Feedback Buttons
                if query_id:
                    submitted_record = st.session_state.feedback_submitted.get(query_id)

                    if submitted_record:
                        st.markdown(
                            f"<small style='color: #34D399; font-weight: 500;'>✅ Feedback saved: <strong>{submitted_record.get('rating')}</strong></small>",
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
                                    "Optional feedback comment:",
                                    placeholder="e.g. Accurate citation or missing detail...",
                                    key=f"comment_input_{query_id}",
                                )
                                submit_btn = st.form_submit_button("Submit Feedback", type="primary")
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
        st.session_state.messages.append({"role": "user", "content": active_prompt})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{active_prompt}**")

        # Call FastAPI backend with LangGraph workflow
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Analyzing question, retrieving from ChromaDB, and grading relevance with LangGraph..."):
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


# View 2: Document Explorer
elif nav_selection == "📁 Document Explorer":
    st.markdown(
        """
        <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.25rem;">Documentation Knowledge Base</h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0;">
                Live inventory of technical documentation indexed in the ChromaDB persistent vector database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

        # Metric Cards (Dark Mode SaaS Glassmorphism)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card-dark">
                    <div class="metric-label-dark">Indexed Documents</div>
                    <div class="metric-val-dark">{total_docs}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card-dark">
                    <div class="metric-label-dark">Total Vector Chunks</div>
                    <div class="metric-val-dark">{total_chunks}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                """
                <div class="metric-card-dark">
                    <div class="metric-label-dark">Vector Store Engine</div>
                    <div class="metric-val-dark" style="font-size: 1.45rem;">ChromaDB <span style="font-size: 0.9rem; color: #94A3B8;">(Cosine)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("### Document Inventory")

        # Search filter
        search_filter = st.text_input("🔍 Filter catalog by title, filename, or extension:", "")

        filtered_docs = [
            d for d in doc_list
            if search_filter.lower() in d.get("source_filename", "").lower()
            or search_filter.lower() in d.get("document_title", "").lower()
            or search_filter.lower() in d.get("file_type", "").lower()
        ]

        if not filtered_docs:
            st.info("No documents found matching your filter criteria.")
        else:
            for d in filtered_docs:
                filename = d.get("source_filename", "unnamed")
                title = d.get("document_title", "Untitled Document")
                file_type = d.get("file_type", ".txt")
                chunk_cnt = d.get("chunk_count", 0)
                char_cnt = d.get("total_characters", 0)
                source_path = d.get("source_path", "")

                # Badge colors based on file format
                badge_style = (
                    "background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3);"
                    if file_type == ".md"
                    else "background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3);"
                    if file_type == ".html"
                    else "background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3);"
                )

                st.markdown(
                    f"""
                    <div class="doc-card-dark">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">{title}</span>
                                <span style="display: inline-block; padding: 0.15rem 0.55rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; margin-left: 0.5rem; {badge_style}">
                                    {file_type.upper()}
                                </span>
                            </div>
                            <span style="font-size: 0.8rem; color: #64748B; font-family: monospace;">📁 {source_path}</span>
                        </div>
                        <div style="margin-top: 0.75rem; display: flex; gap: 1.75rem; font-size: 0.86rem; color: #94A3B8;">
                            <span>📄 Source: <code style="color: #38BDF8;">{filename}</code></span>
                            <span>🧩 Chunks: <strong style="color: #2DD4BF;">{chunk_cnt}</strong></span>
                            <span>📝 Characters: <strong style="color: #E2E8F0;">{char_cnt:,}</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.error(f"Failed to load documents: {doc_resp.get('error')}")


# View 3: Document Ingestion
elif nav_selection == "📤 Upload Documents":
    st.markdown(
        """
        <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.25rem;">Upload & Ingest Documentation</h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0;">
                Expand the RAG knowledge base with new technical guides. Upload Markdown (<code>.md</code>),
                Plain Text (<code>.txt</code>), or HTML (<code>.html</code>) files for automatic semantic chunking,
                vector embedding, and duplicate-safe ChromaDB indexing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop documentation files here",
        type=["md", "txt", "html"],
        accept_multiple_files=True,
        help="Upload Markdown (.md), Text (.txt), or HTML (.html) files.",
    )

    if uploaded_files:
        st.markdown(f"**Selected {len(uploaded_files)} file(s) for ingestion:**")
        for f in uploaded_files:
            st.markdown(f"- 📄 `{f.name}` <span style='color: #64748B;'>({len(f.getvalue()):,} bytes)</span>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Process & Ingest Into Vector Store", type="primary"):
            prepared = []
            for f in uploaded_files:
                file_bytes = f.getvalue()
                ext = os.path.splitext(f.name)[1].lower()
                ctype = "text/markdown" if ext == ".md" else "text/html" if ext == ".html" else "text/plain"
                prepared.append((f.name, file_bytes, ctype))

            with st.spinner("Ingesting, generating semantic chunks, and indexing into ChromaDB..."):
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
