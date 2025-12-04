from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


@st.cache_data(show_spinner=False)
def fetch_collections() -> List[str]:
    try:
        response = requests.get(_api_url("/collections"), timeout=30)
        response.raise_for_status()
        return response.json().get("collections", [])
    except requests.RequestException:
        return []


def ingest_document(payload: Dict) -> Optional[Dict]:
    try:
        response = requests.post(_api_url("/ingest"), json=payload, timeout=1800)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Ingestion failed: {exc}")
        if exc.response is not None:
            st.text(exc.response.text)
        return None


def query_documents(payload: Dict) -> Optional[Dict]:
    try:
        response = requests.post(_api_url("/query"), json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Query failed: {exc}")
        if exc.response is not None:
            st.text(exc.response.text)
        return None


def main():
    st.set_page_config(page_title="Personal Assist", layout="wide")
    st.title("Personal Assist – Document QA")
    st.caption(f"Backend: {API_BASE_URL}")

    tab_ingest, tab_query = st.tabs(["📄 Ingestion", "❓ Question Answering"])

    with tab_ingest:
        st.subheader("Ingest a new PDF")
        uploaded_file = st.file_uploader(
            "Upload PDF (stored temporarily for backend access)",
            type=["pdf"],
            help=f"Files are written to {UPLOAD_DIR}. The backend reads the saved file directly.",
        )
        manual_path = st.text_input(
            "Or provide a backend-accessible path (optional)",
            help="Use this if the file already exists on the backend host.",
        )
        collection_name = st.text_input("Optional collection name override")
        fast_mode = st.toggle("Fast mode", value=True)
        max_pages = st.number_input(
            "Max pages (0 = all)", min_value=0, step=1, value=0, help="Use for quick smoke tests."
        )

        if st.button("Ingest Document", type="primary"):
            resolved_path = None
            if uploaded_file is not None:
                resolved_path = save_uploaded_file(uploaded_file)
                st.info(f"Uploaded file saved to {resolved_path}")
            elif manual_path.strip():
                resolved_path = manual_path.strip()
            else:
                st.warning("Upload a PDF or provide an existing backend-accessible path.")
                resolved_path = None

            if resolved_path:
                payload = {
                    "file_path": resolved_path,
                    "collection_name": collection_name or None,
                    "fast_mode": fast_mode,
                    "max_pages": max_pages or None,
                }
                with st.spinner("Running ingestion pipeline..."):
                    result = ingest_document(payload)
                if result:
                    st.success("Document ingested successfully!")
                    st.json(result)
                    fetch_collections.clear()  # invalidate cache

    with tab_query:
        st.subheader("Ask a question")
        collections = fetch_collections()

        settings_container = st.container()
        with settings_container:
            st.markdown("#### Retrieval settings")
            cols = st.columns(3)
            with cols[0]:
                search_all = st.toggle("Search all collections", value=False)
            with cols[1]:
                use_cache = st.toggle("Use semantic cache", value=True)
            with cols[2]:
                top_k = st.slider("Top documents", min_value=1, max_value=50, value=10)

            if search_all:
                selected_collection = None
                st.info(
                    "Query will run across all available collections in parallel. "
                    "This may take longer depending on how many datasets you have ingested."
                )
            else:
                if not collections:
                    st.warning("No collections found. Ingest a document first.")
                selected_collection = st.selectbox(
                    "Collection", options=collections, index=0 if collections else None
                )

        st.divider()
        st.markdown("#### Chat")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        clear_col1, clear_col2 = st.columns([1, 4])
        with clear_col1:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()

        user_query = st.chat_input("Ask a question")

        if user_query:
            if not search_all and not selected_collection:
                st.warning("Select a collection or enable search across all collections.")
            else:
                payload = {
                    "query": user_query.strip(),
                    "collection_name": selected_collection,
                    "top_k": top_k,
                    "use_cache": use_cache,
                    "search_all_collections": search_all,
                }
                with st.spinner("Retrieving and generating response..."):
                    result = query_documents(payload)
                if result:
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    answer_md = (
                        f"**Collection:** {result['collection_name']}\n\n"
                        f"**Latency:** {result['latency_seconds']:.2f}s | "
                        f"**Cache hit:** {'✅' if result['cache_hit'] else '❌'}\n\n"
                        f"{result['answer']}"
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_md})
                    st.rerun()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    stem = Path(uploaded_file.name).stem or "upload"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:40] or "upload"
    filename = f"{safe_stem}_{uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / filename
    destination.write_bytes(uploaded_file.getbuffer())
    return str(destination)


if __name__ == "__main__":
    main()
