# Personal Assist API

This project exposes the PDF ingestion and hybrid dense+sparse retrieval workflow (born in `full_pipeline.ipynb`) through a FastAPI service backed by Qdrant and Ollama.

## Requirements

Install dependencies locally:

```bash
pip install -r requirements.txt
```

Optional heavy deps (Docling, transformers, GPU extras) are already listed; drop what you do not need.

Copy `.env.example` to `.env` before running anything—the backend calls `load_dotenv()` at startup so it automatically picks up that file.  
`UPLOAD_DIR` controls where uploaded PDFs are stored temporarily so both the Streamlit frontend and the FastAPI backend can read them (defaults to `/app/uploads`). When using Docker, the `./uploads` directory on the host is mounted into both containers.  
`OPEN_MODE` (default `true`) keeps the current HuggingFace embedding + Ollama LLM stack. Set it to `false` to switch to Gemini embeddings/LLM (requires `GEMINI_API_KEY`, `GEMINI_EMBEDDING_MODEL`, and `GEMINI_LLM_MODEL`).

> ⚠️ **Performance disclaimer:** The ingestion pipeline is CPU-heavy (OCR, chunking, embeddings). Processing speed scales with your hardware—more CPU threads, faster disks, and GPUs will dramatically reduce runtime, while smaller machines will see longer ingest/query times.

## Tech Stack

- **FastAPI** for the HTTP API with request validation and async background execution.
- **LangChain** (+ Flashrank, HuggingFace/Gemini embeddings, LangGraph-ready components) for chunking, hybrid retrieval, and LLM orchestration.
- **Qdrant** as the dense+sparse vector store with hybrid retrieval and cache-friendly collection management.
- **Ollama** hosting the default LLM (swappable via `LLM_*` env vars), automatically pulled in Docker.
- **Docling / PyPDF / Tesseract / pdf2image** for high-fidelity PDF parsing, OCR, and quality scoring.
- **Streamlit** for the upload-aware ingestion UI and ChatGPT-style QA interface.
- **Docker & Docker Compose** to run the API + Qdrant + Ollama + Streamlit stack locally with persistent volumes.

## Architecture Overview

1. **Document processing** (`backend/document_processing/`): inspects PDF operators, extracts text with PyPDF when possible, escalates to Tesseract/Docling OCR, and attaches quality metrics before emitting normalized page records.
2. **Semantic chunking** (`backend/embeddings/`): feeds pages into LangChain’s `SemanticChunker`, backed by HuggingFace embeddings in open mode or Gemini embeddings when `OPEN_MODE=false`; chunks keep metadata like source path, page numbers, and extraction method.
3. **Vector database** (`backend/vectorstores/`): provisions Qdrant collections per document with dense + sparse vectors for hybrid retrieval, guards against duplicate collection names, and surfaces `/collections` for discovery.
4. **Pipelines** (`backend/pipelines/`): ingestion orchestrates processing → chunking → Qdrant ingestion; QA pipelines combine Flashrank reranking, contextual compression retrievers, and a semantic cache to shortcut similar queries.
5. **Retrieval + LLM** (`backend/retrieval/`, `main.py`): LangChain chains format context, call either Ollama or Gemini chat models, and return answers with latency/cache diagnostics; `/query` supports single-collection or parallel multi-collection search.
6. **Frontend** (`frontend/app.py`): Streamlit handles file uploads (saved to `UPLOAD_DIR`), exposes ingestion parameters, and provides a chat interface with persistent history and adjustable retrieval settings.

## Environment

Duplicate `.env` and edit the values for your setup (Qdrant URL/API key, embedding model/device, Ollama model/base URL, etc.).  
The API loads this file automatically via `python-dotenv`.

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `POST /ingest` (Body `{ "file_path": "...", "collection_name": "optional-name" }`)
  Runs PDF processing -> semantic chunking -> embedding -> Qdrant ingest. Duplicate collection names trigger `409 Conflict`.

- `POST /query` (Body `{ "query": "...", "collection_name": "optional-name", "use_cache": true, "search_all_collections": false }`)
  Performs hybrid dense+sparse retrieval (Flashrank rerank) and sends the context to the configured LLM. Set `search_all_collections=true` to query every collection in parallel and merge the results.

- `GET /collections`
  Lists every Qdrant collection so clients can choose which dataset to query.

- `GET /health`
  Basic readiness plus last-selected collection and collection count.

## Docker Compose

Spin up the entire stack (FastAPI API, Qdrant with persistent storage, and Ollama with model auto-pull) via:

```bash
docker compose up --build -d
```

What you get:

- API → `http://localhost:8000`
- Streamlit UI → `http://localhost:8501`
- Qdrant → `http://localhost:6333` with data persisted under `./qdrant_storage`
- Ollama → `http://localhost:11434` with models persisted in the `ollama_data` Docker volume
- Shared uploads folder → host `./uploads` mounted as `/app/uploads` inside API + frontend containers

The helper service `ollama-init` automatically pulls the model referenced by `LLM_MODEL` (falls back to `llama3.2`) before the API starts. Adjust `.env` before running to pick a different model, tweak PDF/embedding settings, etc.

## Streamlit Frontend

A lightweight UI lives in `frontend/app.py`. Launch it (while the API is running) with:

```bash
streamlit run frontend/app.py
```

Environment variable `API_BASE_URL` controls which backend instance the UI calls (defaults to `http://localhost:8000`; in Docker Compose it is auto-set to the internal API service).  
The ingestion form now supports direct PDF uploads: files are saved into `UPLOAD_DIR` (shared with the API) before calling `/ingest`, and you can still provide a manual path if needed. The query form can target a specific collection or search across all collections in parallel.

## Customizing

- `PDF_*` env vars control DPI, workers, fast mode, and page caps. Set `PDF_MAX_WORKERS` no higher than your CPU thread count to avoid thrashing, and raise `PDF_DPI` when you need higher-fidelity OCR (at the cost of slower processing and more memory).
- `CHUNK_THRESHOLD_*` tweaks the semantic chunker.
- `DEFAULT_COLLECTION_PREFIX` changes automatic collection slugs (derived from filenames).
- `CACHE_MAX_SIZE` / `CACHE_SIMILARITY` tune the semantic cache.
- `LLM_*` picks the provider/model/base URL; with Docker compose it defaults to the in-network Ollama service.
- `OPEN_MODE=false` swaps the pipeline to Gemini embeddings/LLM; configure the Gemini API key and model names via `GEMINI_*` variables.
- `UPLOAD_DIR` defines where temporary uploads live (ensure the directory exists and is mounted/shared if you run multiple services).

> ⚠️ **Gemini quota note:** Google’s Gemini free tier enforces tight rate/usage limits. If you run with `OPEN_MODE=false` using the free tier, you may encounter quota errors during large ingestions or multi-query chat sessions—upgrade your plan or throttle usage accordingly.

## Development Notes

- All core logic lives in `backend/` (document processing, embeddings, vector stores, retrieval, pipelines, helpers).
- `utils.py` re-exports the most common helpers for notebooks/experiments.
- `main.py` initializes shared resources at startup (embeddings, Qdrant client, Ollama LLM, semantic cache) and exposes ingestion/query/list/health endpoints.

## TODO

- Add query routing for simple questions
- Add chat memory
- Add user authentication and user sessions
- Add voice capabilities (STT → RAG pipeline → TTS)
