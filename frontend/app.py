from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


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
        response = requests.post(_api_url("/ingest"), json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Ingestion failed: {exc}")
        if exc.response is not None:
            st.text(exc.response.text)
        return None


def query_documents(payload: Dict) -> Optional[Dict]:
    try:
        response = requests.post(_api_url("/query"), json=payload, timeout=120)
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
        file_path = st.text_input("PDF file path (accessible to the backend)")
        collection_name = st.text_input("Optional collection name override")
        fast_mode = st.toggle("Fast mode", value=True)
        max_pages = st.number_input(
            "Max pages (0 = all)", min_value=0, step=1, value=0, help="Use for quick smoke tests."
        )

        if st.button("Ingest Document", type="primary"):
            if not file_path:
                st.warning("Please provide a file path.")
            else:
                payload = {
                    "file_path": file_path,
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
        search_all = st.toggle("Search all collections", value=False)
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

        query = st.text_area("Your question")
        top_k = st.slider("Top documents to consider", min_value=1, max_value=50, value=10)
        use_cache = st.toggle("Use semantic cache", value=True)

        if st.button("Run Query", type="primary"):
            if not query.strip():
                st.warning("Please enter a question.")
            elif not search_all and not selected_collection:
                st.warning("Select a collection or enable search across all collections.")
            else:
                payload = {
                    "query": query.strip(),
                    "collection_name": selected_collection,
                    "top_k": top_k,
                    "use_cache": use_cache,
                    "search_all_collections": search_all,
                }
                with st.spinner("Retrieving and generating response..."):
                    result = query_documents(payload)
                if result:
                    st.success("Answer ready!")
                    st.markdown(f"**Collection:** {result['collection_name']}")
                    st.markdown(f"**Latency:** {result['latency_seconds']:.2f}s")
                    st.markdown(f"**Cache hit:** {'✅' if result['cache_hit'] else '❌'}")
                    st.markdown("### Answer")
                    st.write(result["answer"])
                    st.markdown("### Raw response")
                    st.json(result)


if __name__ == "__main__":
    main()
