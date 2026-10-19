import base64
import io
import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Multimodal RAG Assistant")
st.caption("Ask questions about your PDFs and inspect retrieved text, tables, and images.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("API connected")
        st.metric("Stored vectors", health.get("vector_count") if health.get("vector_count") is not None else "Unknown")
        st.caption(f"Model: {health.get('model', 'Unknown')}")
    except requests.RequestException:
        st.error("Cannot connect to FastAPI. Start the API server first.")

    st.subheader("Add PDF documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload and index PDFs", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Parsing PDFs, summarizing multimodal chunks, and building the vector store..."):
            try:
                files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                response = requests.post(f"{API_URL}/ingest", files=files, timeout=3600)
                response.raise_for_status()
                st.success(f"Ingestion complete. Vectors: {response.json().get('vector_count', 'unknown')}")
            except requests.RequestException as exc:
                st.error("PDF ingestion failed.")

                response = getattr(exc, "response", None)

                if response is not None:
                    st.code(
                        f"Status: {response.status_code}\n"
                        f"Response: {response.text}"
                    )
                else:
                    st.code(str(exc))

    show_sources = st.toggle("Show retrieved chunks", value=True)
    show_debug = st.toggle("Show retrieval diagnostics", value=True)
    if st.button("Clear conversation", use_container_width=True):
        if st.session_state.session_id:
            try:
                requests.post(f"{API_URL}/clear", json={"session_id": st.session_state.session_id}, timeout=10)
            except requests.RequestException:
                pass
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()


def display_source(source: dict):
    title = f"Chunk {source.get('chunk_id', '?')}"
    metadata = source.get("metadata", {})
    if metadata.get("filename") or metadata.get("source"):
        title += f" — {metadata.get('filename') or metadata.get('source')}"
    with st.expander(title, expanded=False):
        st.markdown("**Retrieved chunk / searchable summary**")
        st.write(source.get("content", ""))
        if source.get("raw_text") and source["raw_text"] != source.get("content"):
            with st.expander("Original extracted text"):
                st.text(source["raw_text"][:12000])
        tables = source.get("tables_html", []) or []
        if tables:
            st.markdown(f"**Tables ({len(tables)})**")
            for table_index, table_html in enumerate(tables, start=1):
                st.caption(f"Table {table_index}")
                rendered = False
                if isinstance(table_html, str) and "<table" in table_html.lower():
                    try:
                        frames = pd.read_html(io.StringIO(table_html))
                        for frame in frames:
                            st.dataframe(frame, use_container_width=True, hide_index=True)
                        rendered = bool(frames)
                    except (ValueError, ImportError):
                        rendered = False
                if not rendered:
                    st.markdown(table_html, unsafe_allow_html=True)
        images = source.get("images_base64", []) or []
        if images:
            st.markdown(f"**Images / diagrams ({len(images)})**")
            for image_index, image_data in enumerate(images, start=1):
                try:
                    encoded = image_data.split(",", 1)[1] if image_data.startswith("data:image") else image_data
                    st.image(base64.b64decode(encoded), caption=f"Extracted image {image_index}", use_container_width=True)
                except Exception:
                    st.warning(f"Could not render extracted image {image_index}; the stored image payload may be incomplete.")
        if metadata:
            with st.expander("Chunk metadata"):
                st.json(metadata)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            result = message["result"]
            if show_debug:
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieved chunks", result.get("retrieved_count", 0))
                c2.metric("Response time", f"{result.get('elapsed_seconds', 0)} s")
                c3.metric("Sources shown", len(result.get("sources", [])))
                with st.expander("Retrieval diagnostics"):
                    st.write("**Rewritten search query:**", result.get("rewritten_question", ""))
            if show_sources:
                for source in result.get("sources", []):
                    display_source(source)

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "session_id": st.session_state.session_id, "show_sources": show_sources},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.session_id = result["session_id"]
                st.markdown(result["answer"])
                if show_debug:
                    c1, c2 = st.columns(2)
                    c1.metric("Retrieved chunks", result.get("retrieved_count", 0))
                    c2.metric("Response time", f"{result.get('elapsed_seconds', 0)} s")
                    with st.expander("Retrieval diagnostics"):
                        st.write("**Rewritten search query:**", result.get("rewritten_question", ""))
                if show_sources:
                    for source in result.get("sources", []):
                        display_source(source)
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
            except requests.RequestException as exc:
                error = f"Request failed: {exc}"
                if getattr(exc, "response", None) is not None:
                    error += f"\n\n{exc.response.text}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
