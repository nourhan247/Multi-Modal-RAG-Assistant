# Multimodal RAG Assistant for PDF Documents

A conversational Retrieval-Augmented Generation (RAG) system that lets you chat with PDFs containing **text, tables, and images**. Answers are grounded in the retrieved content, and the original text, tables, and images behind every answer are shown next to it so you can verify the source.

Built with Python, LangChain, Unstructured, ChromaDB, Groq, FastAPI, and Streamlit.

## Features

- **Multimodal PDF parsing**: extracts text, tables (as structured HTML), and images using Unstructured's hi-res strategy with Tesseract OCR.
- **Smart chunking**: groups content by document title and section so related content stays together.
- **Searchable summaries for tables and images**: chunks that contain tables or images are summarized by an LLM into a detailed, searchable description. The original text, table HTML, and base64 images are stored in the chunk metadata and returned with the answer.
- **Incremental ingestion**: chunk IDs are derived from a SHA-256 hash of the file plus the chunk index, so re-uploading the same PDF does not create duplicate vectors.
- **Conversational retrieval**: follow-up questions are rewritten into standalone search queries using the last 5 turns of chat history.
- **Grounded answers**: the model is instructed to answer only from retrieved chunks, treat them as untrusted data (not instructions), and say so when the documents don't contain enough information.
- **Source inspection**: the Streamlit UI shows retrieved chunks, extracted tables, images, metadata, the rewritten search query, and response time.

## How it works

```
PDF ──► Unstructured (hi-res + OCR) ──► chunk by title
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │ text-only chunk                               │ chunk with tables / images
                    │ (stored as is)                                │ (LLM writes a searchable summary)
                    └───────────────────────┬───────────────────────┘
                                            ▼
                     Sentence-Transformers embeddings ──► ChromaDB (cosine)

Question ──► rewrite with chat history ──► top-k retrieval ──► grounded LLM answer
                                                                + original text, tables, images
```

## Tech stack

| Area | Tools |
|---|---|
| Document parsing | Unstructured, Tesseract OCR (pytesseract) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (cosine similarity) |
| LLM | Qwen on Groq via LangChain (configurable with `GROQ_MODEL`) |
| Backend | FastAPI |
| Frontend | Streamlit |

## Project structure

```
├── multi_rag.py       # Ingestion: parse PDFs, chunk, summarize, embed, store
├── retrieval2.py      # Command-line chat over the vector store
├── api.py             # FastAPI backend (chat, ingest, clear, health)
├── streamlit_app.py   # Streamlit web interface
├── data/              # PDFs to ingest
└── dbv2/chroma_db/    # Persisted vector store (created automatically)
```

## Getting started

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed. `multi_rag.py` currently points to the default Windows path (`C:\Program Files\Tesseract-OCR\tesseract.exe`); change it if yours is elsewhere.
- Poppler, which Unstructured uses for PDF processing
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install langchain-chroma langchain-huggingface langchain-groq langchain-core \
            "unstructured[pdf]" pytesseract fastapi uvicorn python-multipart \
            streamlit pandas requests python-dotenv
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
# Optional
# GROQ_MODEL=your_model_name
# RETRIEVER_K=3
```

## Usage

### Option 1: Web app

Start the API:

```bash
uvicorn api:app --reload
```

In a second terminal, start the interface:

```bash
streamlit run streamlit_app.py
```

Then upload PDFs from the sidebar ("Upload and index PDFs") and start asking questions.

### Option 2: Command line

Put PDFs in the `data/` folder, then:

```bash
python multi_rag.py      # ingest all PDFs in data/
python retrieval2.py     # chat in the terminal ("clear" resets history, "quit" exits)
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status, vector store status, vector count, model name |
| POST | `/chat` | Ask a question; returns the answer, rewritten query, and retrieved sources |
| POST | `/ingest` | Upload PDFs and add them to the vector store |
| POST | `/clear` | Clear the conversation history for a session |

Example request:

```json
POST /chat
{
  "question": "What does the table on revenue show?",
  "session_id": "optional-session-id",
  "show_sources": true
}
```

## Configuration

| Setting | Default | Where |
|---|---|---|
| `GROQ_MODEL` | Qwen model on Groq | `.env` |
| `RETRIEVER_K` | `3` | `.env` |
| `MAX_HISTORY_TURNS` | `5` | `api.py`, `retrieval2.py` |
| Chunk size | 3,000 chars max (soft limit 2,400) | `multi_rag.py` |
| `RAG_API_URL` | `http://127.0.0.1:8000` | Streamlit environment variable |

## Possible improvements

- Reranking of retrieved chunks
- Evaluation set for retrieval and answer quality
- Source citations with page numbers
- Persistent chat history instead of in-memory sessions
