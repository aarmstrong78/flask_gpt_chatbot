# DocuChat

DocuChat is a Flask web application that provides a simple interface for chatting with OpenAI models. It allows you to upload PDF, DOCX, or TXT files so the chatbot can reference the content of your documents when generating responses.

## Features

- Upload documents (PDF, DOCX, TXT) and create embeddings using FAISS for retrieval.
- Chat interface with streaming responses from the OpenAI API.
- Manage uploaded documents and delete sources when no longer needed.
- Basic error handling pages for 404 and 500 errors.
- Dockerfile for containerized deployment.

## Requirements

- Python 3.9+
- An OpenAI API key

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file or set the following environment variables:

- `OPENAI_API_KEY` – your OpenAI API key
- `SECRET_KEY` – secret key used by Flask

## Running

Start the development server:

```bash
python app.py
```

For production you can use Gunicorn (as used in the Dockerfile):

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

## Docker

Build and run with Docker:

```bash
docker build -t docuchat .
docker run -p 5000:5000 -e OPENAI_API_KEY=... -e SECRET_KEY=... docuchat
```

Then visit `http://localhost:5000` to interact with the chat interface.

## License

This project is provided as-is without any warranty.
