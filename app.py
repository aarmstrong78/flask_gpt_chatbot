"""Main application module for the Flask GPT chatbot.

This file wires together the Flask routes, LangChain components and vector
store logic used to provide a conversational interface backed by OpenAI models.
"""

# app.py

import os
import openai
from openai import RateLimitError

from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from authlib.integrations.flask_client import OAuth

from langchain.chains import ConversationChain
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss

from cachetools import TTLCache, cached

import json
import uuid  # Ensure uuid is imported


# =====================
# Logging Configuration
# =====================
import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Define the log message format
    handlers=[
        logging.StreamHandler(),  # Log messages will be output to the console
        logging.FileHandler("app.log"),  # Log messages will also be saved to app.log
    ],
)
# Create a logger instance
logger = logging.getLogger(__name__)

# Log that the application has started
logger.info("Starting Flask GPT Chatbot Application.")

# =====================
# Environment Variables
# =====================

# Load environment variables from .env file
load_dotenv()

# =====================
# Flask Application
# =====================

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

# User session management
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url="https://oauth2.googleapis.com/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    api_base_url="https://www.googleapis.com/oauth2/v1/",
    client_kwargs={"scope": "openid email profile"},
)


class User(UserMixin):
    """Simple user model for session handling."""

    def __init__(self, user_id, name, email):
        self.id = user_id
        self.name = name
        self.email = email


users = {}
user_states = {}


@login_manager.user_loader
def load_user(user_id):
    """Return a user object by ID."""
    return users.get(user_id)


# Configure upload settings
UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ``data`` holds the FAISS index and other persisted information
os.makedirs("data", exist_ok=True)

# =====================
# Caching Configuration
# =====================

# Initialize an in-memory cache for embeddings. This avoids repeatedly
# generating embeddings for the same text within a short period.
cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour TTL

# =====================
# Helper Functions
# =====================


# Initialize OpenAI API key and validate presence
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # Fail fast if the API key is missing so the user knows to configure it
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

# Configure the OpenAI SDK with the provided key
openai.api_key = openai_api_key


def allowed_file(filename):
    """Return ``True`` if the file has an allowed extension."""

    # Ensure there is a file extension and check it against the whitelist
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path, filename):
    """Load the uploaded document and split it into text chunks."""

    # Choose the appropriate loader based on file extension
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    else:
        # Prevent processing of unsupported file types
        raise ValueError("Unsupported file type")

    # Load the raw document data from disk
    documents = loader.load()

    # Break the document into smaller chunks so embeddings are more manageable
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # Log the extracted texts
    for idx, text in enumerate(texts):
        logger.info(f"Text chunk {idx + 1}: {text.page_content[:100]}...")  # Log first 100 chars

    # Return the list of Document objects generated from the uploaded file
    return texts


# Initialize the language model shared across users
llm = ChatOpenAI(
    temperature=1,  # GPT5 does not support temp<>1
    model_name="gpt-5",
    streaming=True,  # Enable streaming so responses can be incrementally sent
    openai_api_key=openai_api_key,
)
logger.info(f"Using OpenAI model: {llm.model_name}")


# Initialize embeddings globally
# Embedding model used to convert text into high-dimensional vectors for FAISS
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Specify the embedding model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=6,  # Increase the number of retries
)


@cached(cache)
def get_cached_embedding(text):
    """Return cached embedding for the given text."""
    return embeddings.embed_query(text)


def cached_similarity_search(query, k=10):
    """Return cached similar documents for the query for the current user."""
    state = get_user_state()
    cache = state["similarity_cache"]
    if query in cache:
        return cache[query]
    if state["vector_store"] is None:
        results = []
    else:
        results = state["vector_store"].similarity_search(query, k=k)
    cache[query] = results
    return results


def get_user_state():
    """Return or create the state object for the authenticated user."""
    user_id = current_user.get_id()
    state = user_states.get(user_id)
    if state is None:
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory)
        vector_store_path = f"data/vector_store_{user_id}.faiss"
        mapping_path = f"data/document_mapping_{user_id}.json"
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    document_mapping = json.load(f)
            except json.JSONDecodeError:
                document_mapping = {}
        else:
            document_mapping = {}
            with open(mapping_path, "w") as f:
                json.dump(document_mapping, f)
        vector_store = None
        if os.path.exists(vector_store_path):
            try:
                vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            except Exception:
                vector_store = None
        state = {
            "memory": memory,
            "conversation": conversation,
            "vector_store": vector_store,
            "document_mapping": document_mapping,
            "vector_store_path": vector_store_path,
            "mapping_path": mapping_path,
            "similarity_cache": TTLCache(maxsize=1000, ttl=600),
        }
        user_states[user_id] = state
    return state


def rebuild_faiss():
    """Rebuild the FAISS vector store for the current user."""
    state = get_user_state()
    try:
        sample_embedding = embeddings.embed_query("test")
        embedding_dim = len(sample_embedding)
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        state["vector_store"] = FAISS(
            embedding_function=embeddings,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        all_documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in state["document_mapping"].values()
        ]
        if all_documents:
            state["vector_store"].add_documents(all_documents)
            logger.info(f"Added {len(all_documents)} documents to FAISS vector store.")
        else:
            logger.info("No documents to add to FAISS vector store.")
        state["vector_store"].save_local(state["vector_store_path"])
        logger.info("FAISS vector store rebuilt and saved to disk.")
    except Exception as e:
        logger.error(f"Failed to rebuild FAISS vector store: {e}")
        state["vector_store"] = None


# =====================
# Core Functions
# =====================


def get_gpt_response(user_input):
    """
    Generate a response from GPT-4 based on user input and contextual embeddings.
    """
    try:
        state = get_user_state()
        if state["vector_store"] is not None:
            relevant_docs = cached_similarity_search(user_input, k=10)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            for idx, doc in enumerate(relevant_docs):
                snippet = (
                    doc.page_content[:100].replace("\n", " ") + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                logger.info(f"Retrieved Doc {idx + 1}: {snippet}")
            state["memory"].chat_memory.add_user_message(f"Context:\n{context}")
        else:
            logger.warning("Vector store is empty. No context available.")

        state["memory"].chat_memory.add_user_message(user_input)
        response = state["conversation"].predict(input=user_input)
        return response
    except RateLimitError as e:
        # Log the error
        logger.error(f"Rate limit exceeded: {e}")
        # Inform the user
        return "I'm currently experiencing high demand. Please try again later."
    except Exception as e:
        # Handle other exceptions
        logger.error(f"An error occurred: {e}")
        return "An unexpected error occurred. Please try again later."


def get_gpt_response_stream(user_input):
    """
    Generate a streaming response from GPT-4o based on user input and contextual embeddings.
    Yields tokens as they are generated.
    """
    try:
        state = get_user_state()
        if state["vector_store"] is not None:
            relevant_docs = cached_similarity_search(user_input, k=10)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            for idx, doc in enumerate(relevant_docs):
                snippet = (
                    doc.page_content[:100].replace("\n", " ") + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                logger.info(f"Retrieved Doc {idx + 1}: {snippet}")
            state["memory"].chat_memory.add_user_message(f"Context:\n{context}")
        else:
            logger.warning("Vector store is empty. No context available.")

        state["memory"].chat_memory.add_user_message(user_input)

        messages = state["memory"].chat_memory.messages
        ai_response = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            if content:
                ai_response += content
                yield content

        state["memory"].chat_memory.add_ai_message(ai_response)

    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        yield "I'm currently experiencing high demand. Please try again later."
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        yield "An unexpected error occurred. Please try again later."


def remove_document(document_id):
    """
    Remove a document and its embeddings from the FAISS vector store and document mapping.
    """
    state = get_user_state()
    document_mapping = state["document_mapping"]
    if document_id not in document_mapping:
        logger.warning(f"Document ID {document_id} not found in mapping.")
        raise ValueError("Document ID not found.")

    source = document_mapping[document_id]["metadata"]["source"]
    to_delete_ids = [doc_id for doc_id, doc in document_mapping.items() if doc["metadata"]["source"] == source]
    for doc_id in to_delete_ids:
        del document_mapping[doc_id]
    logger.info(f"Removed {len(to_delete_ids)} documents associated with source '{source}' from document mapping.")

    with open(state["mapping_path"], "w") as f:
        json.dump(document_mapping, f)

    logger.info("Rebuilding FAISS vector store to reflect deletions.")
    rebuild_faiss()


@app.route("/login")
def login():
    """Initiate Google OAuth login."""
    redirect_uri = url_for("authorize", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/authorize")
def authorize():
    """Handle the OAuth callback from Google."""
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token)
    user = User(user_info["sub"], user_info.get("name", ""), user_info.get("email", ""))
    users[user.id] = user
    login_user(user)
    get_user_state()
    return redirect(url_for("chat"))


@app.route("/logout")
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
def home():
    """Redirect to chat if authenticated, otherwise to login."""
    if current_user.is_authenticated:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/upload_page", methods=["GET"])
@login_required
def upload_page():
    """Render the upload page via a GET request."""
    state = get_user_state()
    unique_sources = list({doc["metadata"]["source"] for doc in state["document_mapping"].values()})
    return render_template("upload.html", sources=unique_sources)


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    """Handle file uploads and update the user's vector store."""
    state = get_user_state()

    if "file" not in request.files:
        flash("No file part")
        logger.warning("No file part in the request.")
        return redirect(url_for("upload_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        logger.warning("No file selected for upload.")
        return redirect(url_for("upload_page"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], current_user.get_id())
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)
        flash("File successfully uploaded")
        logger.info(f"File uploaded: {filename}")

        try:
            texts = extract_text(file_path, filename)
            logger.info(f"Extracted {len(texts)} text chunks from {filename}.")
        except ValueError as ve:
            flash(str(ve))
            logger.error(f"Error extracting text: {ve}")
            return redirect(url_for("upload_page"))

        if texts:
            try:
                if state["vector_store"] is None:
                    rebuild_faiss()
                    if state["vector_store"] is None:
                        flash("Failed to initialize the vector store. Please check the logs.")
                        return redirect(url_for("upload_page"))

                source_id = filename
                new_documents = [
                    Document(page_content=text.page_content, metadata={"source": source_id}) for text in texts
                ]

                if state["vector_store"] is not None:
                    state["vector_store"].add_documents(new_documents)
                    logger.info(f"Added {len(new_documents)} new documents to FAISS vector store.")

                for doc in new_documents:
                    document_id = str(uuid.uuid4())
                    state["document_mapping"][document_id] = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "file_path": file_path,
                    }

                with open(state["mapping_path"], "w") as f:
                    json.dump(state["document_mapping"], f)

                if state["vector_store"] is not None:
                    state["vector_store"].save_local(state["vector_store_path"])
                    logger.info("FAISS vector store saved to disk after upload.")

                flash("File content integrated into context")
            except RateLimitError as e:
                flash("Rate limit exceeded while processing your file. Please try again later.")
                logger.error(f"Rate limit exceeded during embedding: {e}")
            except Exception as e:
                flash("An error occurred while processing your file. Please try again.")
                logger.error(f"Error during embedding or saving vector store: {e}")
        else:
            flash("No text extracted from the uploaded file")
            logger.warning("No text extracted from the uploaded file.")

        return redirect(url_for("chat"))
    flash("Allowed file types are pdf, docx, txt")
    logger.warning(f"Attempted to upload unsupported file type: {file.filename}")
    return redirect(url_for("upload_page"))


@app.route("/chat", methods=["GET"])
@login_required
def chat():
    """Render the chat interface."""
    state = get_user_state()
    return render_template("chat.html", document_mapping=state["document_mapping"])


@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get GPT response
    response = get_gpt_response(user_input)

    return jsonify({"response": response})


@app.route("/stream_response", methods=["POST"])
@login_required
def stream_response():
    """
    Handle chat messages and stream responses from GPT-4o.
    Expects JSON data with 'message'.
    """
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Create a generator for the streaming response
    response_generator = get_gpt_response_stream(user_input)

    # Return a streaming response using Flask's Response
    return Response(response_generator, mimetype="text/plain")


@app.route("/remove_document", methods=["POST"])
@login_required
def remove_document_route():
    """
    Endpoint to remove a document and its embeddings from the context.
    Expects JSON data with 'document_id'.
    """
    data = request.get_json()
    document_id = data.get("document_id")

    if not document_id:
        return jsonify({"error": "No document_id provided"}), 400

    try:
        remove_document(document_id)
        flash(f"Document {document_id} removed successfully.")
        return (
            jsonify({"message": f"Document {document_id} removed successfully."}),
            200,
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error removing document {document_id}: {e}")
        return jsonify({"error": "Failed to remove document."}), 500


@app.route("/delete_sources", methods=["POST"])
@login_required
def delete_sources():
    """Handle deletion of selected sources for the current user."""
    state = get_user_state()
    selected_sources = request.form.getlist("sources")

    if not selected_sources:
        flash("No sources selected for deletion.")
        logger.warning("No sources selected for deletion.")
        return redirect(url_for("upload_page"))

    try:
        with open(state["mapping_path"], "w") as f:
            to_delete_ids = [
                doc_id
                for doc_id, doc in state["document_mapping"].items()
                if doc["metadata"]["source"] in selected_sources
            ]
            for doc_id in to_delete_ids:
                del state["document_mapping"][doc_id]
            json.dump(state["document_mapping"], f)

        state["vector_store"] = None
        rebuild_faiss()

        flash("Selected sources have been successfully deleted.")
        return redirect(url_for("upload_page"))
    except Exception as e:
        logger.error(f"Error deleting selected sources: {e}")
        flash("An error occurred while deleting selected sources. Please try again.")
        return redirect(url_for("upload_page"))


# =====================
# Error Handlers
# =====================


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 Internal Server Errors.
    """
    logger.error(f"Server Error: {error}")
    return render_template("500.html"), 500


@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 Not Found Errors.
    """
    logger.error(f"Not Found Error: {error}")
    return render_template("404.html"), 404


# =====================
# Run the Application
# =====================

if __name__ == "__main__":
    app.run(debug=True)
