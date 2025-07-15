# Technical Overview

This document describes the architecture and key components of **DocuChat**, a Flask application that
provides chat capabilities backed by OpenAI models and context-aware document retrieval.

## Application Structure

```
flask_gpt_chatbot/
├── app.py                # Main Flask application
├── templates/            # HTML templates
├── static/               # CSS and static assets
├── tests/                # Pytest test suite
├── Dockerfile            # Container build file
├── requirements.txt      # Python dependencies
└── docs/TECHNICAL_OVERVIEW.md  # This documentation
```

### Key Paths

- **Uploads**: `uploads/` – temporary location for user provided files.
- **Vector store**: `data/vector_store.faiss` – FAISS index of document embeddings.
- **Document mapping**: `data/document_mapping.json` – tracks metadata for uploaded text.
- **Record manager**: `data/record_manager_cache.sql` – SQLite database used by LangChain's
  `SQLRecordManager`.

## Environment Variables

The application requires the following variables:

| Variable | Description |
| -------- | ----------- |
| `OPENAI_API_KEY` | API key used by `openai` and LangChain components. |
| `SECRET_KEY` | Flask secret key for session management. |

Variables can be provided in a `.env` file or via the environment before running the server.

## Request Flow

1. **Upload documents** via `/upload_page`.
2. `app.py` extracts text using `PyPDFLoader`, `UnstructuredWordDocumentLoader` or `TextLoader` depending
   on file type.
3. Extracted text is split and embedded with OpenAI embeddings. The resulting vectors are stored in FAISS
   along with mapping entries in `document_mapping.json`.
4. Chat messages are sent to `/stream_response` which streams tokens from `ChatOpenAI`. Prior to calling
the model, relevant context is retrieved from FAISS using `cached_similarity_search` to incorporate
information from uploaded documents.
5. Responses are returned to the browser using a streaming HTTP response and rendered incrementally via
JavaScript in `templates/chat.html`.

## Caching

Embeddings and similarity results are cached using [`cachetools.TTLCache`](https://cachetools.readthedocs.io/).
The default cache sizes and TTLs are defined near the top of `app.py`:

```python
cache = TTLCache(maxsize=1000, ttl=3600)        # Embeddings
similarity_cache = TTLCache(maxsize=1000, ttl=600)  # Similarity searches
```

## Document Management

Uploaded documents are tracked in `document_mapping.json`.
Entries include both the page content and metadata such as source filename.
The routes `/remove_document` and `/delete_sources` allow removing individual records or entire source files.
After deletion the FAISS store is rebuilt using `rebuild_faiss()` to keep embeddings in sync.

## Tests

The `tests/` directory contains a Pytest suite focused on error handling. To run tests and style checks:

```bash
black --check .
flake8
pytest -q
```

## Docker Usage

A `Dockerfile` is provided to deploy the application with Gunicorn:

```bash
docker build -t docuchat .
docker run -p 5000:5000 -e OPENAI_API_KEY=... -e SECRET_KEY=... docuchat
```

The container exposes the application on port `5000`.

