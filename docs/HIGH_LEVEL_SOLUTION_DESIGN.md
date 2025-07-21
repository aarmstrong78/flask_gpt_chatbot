# High Level Solution Design

This document outlines the overall architecture and operational approach for **DocuChat**. It summarizes the requirements, high level design decisions, libraries and key functions involved, as well as networking, RBAC, security considerations and testing strategy.

## Requirements

* Python 3.9 or later
* An OpenAI API key (`OPENAI_API_KEY`)
* Flask secret key for sessions (`SECRET_KEY`)
* Ability to store temporary uploads and a FAISS index under the `data/` directory

## Solution Overview

DocuChat is a single container Flask web application that accepts document uploads and allows users to chat with an OpenAI model using the uploaded content as additional context. Documents are parsed and chunked, embedded using `OpenAIEmbeddings` and stored in a local FAISS index. User queries are enriched with the most relevant text before calling the LLM. Responses are streamed back to the browser.

The core flow is:

1. `/upload` receives files and persists metadata in `document_mapping.json`.
2. Embeddings are generated and stored in `data/vector_store.faiss`.
3. `/stream_response` and `/get_response` obtain messages from the client and call `get_gpt_response_stream` or `get_gpt_response` to interact with the model.
4. Documents can be removed through `/remove_document` or `/delete_sources`, which rebuild the vector store.

## Libraries Used

* **Flask** – web framework providing routing and templating
* **LangChain** – conversation handling, embedding and vector store abstractions
* **FAISS** – similarity search over document embeddings
* **OpenAI SDK** – access to ChatGPT and embedding models
* **cachetools** – TTL caches for embeddings and similarity search
* **pytest**, **flake8**, **black** – testing and linting

## Key Functions

* `extract_text` – determines loader by file type and splits content into chunks
* `get_gpt_response` – retrieves documents and produces a single LLM response
* `get_gpt_response_stream` – streaming variant yielding tokens as generated
* `remove_document` – deletes a document from mappings and rebuilds FAISS
* `rebuild_faiss` – regenerates the FAISS index from persisted mappings

## Networking Design

The Flask server listens on port `5000` by default. The browser communicates over HTTP using JSON for chat messages. Long responses are streamed using a `text/plain` streaming endpoint so the UI can display tokens progressively. All network traffic occurs within the container or between the user’s browser and the server; there are no additional internal services.

## RBAC

DocuChat has a simple security model. There are no distinct user roles beyond standard Flask session handling. All authenticated users (the application assumes authentication occurs externally if needed) can upload documents and initiate chats. For deployments requiring stricter control, front-end or reverse proxy authentication can be layered on top of the application.

## Security and Controls

* File uploads are restricted to PDF, DOCX and TXT extensions
* Documents are stored in an isolated `uploads/` directory
* API keys and secrets are loaded from environment variables, not hard coded
* The FAISS index and document metadata reside under `data/` which can be secured with standard filesystem permissions
* Errors are logged and returned via Flask error handlers to avoid leaking stack traces

## Testing

The repository includes a small pytest suite focused on error handling. Developers should run the following prior to submitting changes:

```bash
black --check .
flake8
pytest -q
```

These checks ensure consistent style and basic functionality.

