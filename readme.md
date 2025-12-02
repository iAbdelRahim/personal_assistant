# Personal Assist API

This project exposes the PDF ingestion and hybrid dense+sparse retrieval workflow (born in `full_pipeline.ipynb`) through a FastAPI service backed by Qdrant and Ollama.

## Requirements

Install dependencies locally:

```bash
pip install -r requirements.txt
```

Optional heavy deps (Docling, transformers, GPU extras) are already listed; drop what you do not need.

## Tech Stack

- **FastAPI** for the HTTP API with request validation and async background execution.
- **LangChain** (+ Flashrank, HuggingFace embeddings, LangGraph-ready components) for chunking, hybrid retrieval, and LLM orchestration.
- **Qdrant** as the dense+sparse vector store with hybrid retrieval and cache-friendly collection management.
- **Ollama** hosting the default LLM (swappable via `LLM_*` env vars), automatically pulled in Docker.
- **Docling / PyPDF / Tesseract / pdf2image** for high-fidelity PDF parsing, OCR, and quality scoring.
- **Docker & Docker Compose** to run the API + Qdrant + Ollama stack locally with persistent volumes.

## Environment

Duplicate `.env` and edit the values for your setup (Qdrant URL/API key, embedding model/device, Ollama model/base URL, etc.).  
The API loads this file automatically via `python-dotenv`.

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `POST /ingest` – Body `{"file_path": "...", "collection_name": "optional-name"}`  
  Executes PDF processing → semantic chunking → embedding → Qdrant ingestion. If the resolved collection already exists the API raises `409 Conflict` to prevent duplicate uploads.

- `POST /query` – Body `{"query": "...", "collection_name": "optional-name", "use_cache": true, "search_all_collections": false}`  
  Runs hybrid retrieval (dense+BM25, Flashrank rerank) and feeds the result to the configured LLM (default: Ollama). When `search_all_collections` is `true`, the API queries every collection in parallel, merges the contexts, and labels the response as coming from “all” collections. Returns answer text plus latency/cache stats.

- `GET /collections` – Lists every Qdrant collection so clients can choose which dataset to query.

- `GET /health` – Basic readiness plus last-selected collection and collection count.

## Docker Compose

Spin up the entire stack (FastAPI API, Qdrant with persistent storage, and Ollama with model auto-pull) via:

```bash
docker compose up --build -d
```

What you get:

- API → `http://localhost:8000`
- Qdrant → `http://localhost:6333` with data persisted under `./qdrant_storage`
- Ollama → `http://localhost:11434` with models persisted in the `ollama_data` Docker volume

The helper service `ollama-init` automatically pulls the model referenced by `LLM_MODEL` (falls back to `llama3.2`) before the API starts. Adjust `.env` before running to pick a different model, tweak PDF/embedding settings, etc.

## Streamlit Frontend

A lightweight UI lives in `frontend/app.py`. Launch it (while the API is running) with:

```bash
streamlit run frontend/app.py
```

Environment variable `API_BASE_URL` controls which backend instance the UI calls (defaults to `http://localhost:8000`).  
The ingestion form expects a PDF path that the backend container/process can reach, and the query form can either target a specific collection or search across all collections in parallel.

## Customizing

- `PDF_*` env vars control DPI, workers, fast mode, and page caps. Set `PDF_MAX_WORKERS` no higher than your CPU thread count to avoid thrashing, and raise `PDF_DPI` when you need higher-fidelity OCR (at the cost of slower processing and more memory).
- `CHUNK_THRESHOLD_*` tweaks the semantic chunker.
- `DEFAULT_COLLECTION_PREFIX` changes automatic collection slugs (derived from filenames).
- `CACHE_MAX_SIZE` / `CACHE_SIMILARITY` tune the semantic cache.
- `LLM_*` picks the provider/model/base URL; with Docker compose it defaults to the in-network Ollama service.

## Development Notes

- All core logic lives in `backend/` (document processing, embeddings, vector stores, retrieval, pipelines, helpers).
- `utils.py` re-exports the most common helpers for notebooks/experiments.
- `main.py` initializes shared resources at startup (embeddings, Qdrant client, Ollama LLM, semantic cache) and exposes ingestion/query/list/health endpoints.

## TODO

- Add query routing for simple questions
- Add chat memory
- Add user authentication and user sessions
- Add voice capabilities (STT → RAG pipeline → TTS)
