# app.py

import os
import openai
from openai import RateLimitError

from flask import Flask, Response, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.indexes import SQLRecordManager
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
app.secret_key = os.getenv("SECRET_KEY")

# Configure upload settings
UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

# =====================
# Caching Configuration
# =====================

# Initialize embedding cache with TTL of 1 hour and max size of 1000
cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour TTL

# =====================
# Helper Functions
# =====================


# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # Log the extracted texts
    for idx, text in enumerate(texts):
        logger.info(f"Text chunk {idx + 1}: {text.page_content[:100]}...")  # Log first 100 chars

    return texts


# Initialize LangChain components globally
memory = ConversationBufferMemory()
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o",
    # model_name="o1-mini",
    streaming=True,  # Enable streaming
)

conversation = ConversationChain(llm=llm, memory=memory)
logger.info(f"Using OpenAI model: {llm.model_name}")


# Define paths for vector store and document mapping
vector_store_path = "data/vector_store.faiss"
mapping_path = "data/document_mapping.json"
record_manager_path = "data/record_manager_cache.sql"

# Initialize RecordManager
namespace = "faiss/convodocs"  # Define a namespace for your application


if os.path.exists(record_manager_path):
    try:
        record_manager = SQLRecordManager(namespace, db_url=f"sqlite:///{record_manager_path}")
        logger.info("Loaded existing SQLRecordManager.")
    except Exception as e:
        logger.error(f"Failed to load SQLRecordManager: {e}")
        record_manager = SQLRecordManager(namespace, db_url=f"sqlite:///{record_manager_path}")
        record_manager.create_schema()
        logger.info("Initialized new SQLRecordManager with schema.")
else:
    record_manager = SQLRecordManager(namespace, db_url=f"sqlite:///{record_manager_path}")
    record_manager.create_schema()
    logger.info("Initialized new SQLRecordManager with schema.")


# Initialize embeddings globally
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Specify the embedding model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=6,  # Increase the number of retries
)


@cached(cache)
def get_cached_embedding(text):
    """Return cached embedding for the given text."""
    return embeddings.embed_query(text)


# =====================
# FAISS Vector Store Initialization
# =====================

# Initialize FAISS vector store as None; it will be loaded or created as needed
vector_store = None


def rebuild_faiss():
    """Rebuild the FAISS vector store from ``document_mapping.json``."""
    global vector_store, document_mapping
    try:
        # Initialize FAISS vector store
        sample_embedding = embeddings.embed_query("test")
        embedding_dim = len(sample_embedding)
        logger.info(f"Embedding dimension determined: {embedding_dim}")

        # Create a FAISS index
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=embeddings, index=faiss_index, docstore=InMemoryDocstore(), index_to_docstore_id={}
        )

        # Load all documents from document_mapping.json
        with open(mapping_path, "r") as f:
            document_mapping = json.load(f)

        all_documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in document_mapping.values()
        ]

        # Add all documents to FAISS
        if all_documents:
            vector_store.add_documents(all_documents)
            logger.info(f"Added {len(all_documents)} documents to FAISS vector store.")
        else:
            logger.info("No documents to add to FAISS vector store.")

        # Save the FAISS vector store to disk
        vector_store.save_local(vector_store_path)
        logger.info("FAISS vector store rebuilt and saved to disk.")
    except Exception as e:
        logger.error(f"Failed to rebuild FAISS vector store: {e}")
        vector_store = None


# =====================
# Document Mapping Initialization
# =====================

# Load or initialize the document mapping
if os.path.exists(mapping_path):
    try:
        with open(mapping_path, "r") as f:
            # Check if the file is empty
            if os.path.getsize(mapping_path) > 0:
                document_mapping = json.load(f)
                logger.info("Loaded existing document mapping.")
            else:
                # File is empty, initialize empty dict
                document_mapping = {}
                with open(mapping_path, "w") as fw:
                    json.dump(document_mapping, fw)
                logger.warning("document_mapping.json was empty. Initialized as empty dictionary.")
    except json.JSONDecodeError:
        # JSON is malformed, initialize empty dict
        document_mapping = {}
        with open(mapping_path, "w") as fw:
            json.dump(document_mapping, fw)
        logger.error("document_mapping.json was malformed. Reinitialized as empty dictionary.")
else:
    document_mapping = {}
    with open(mapping_path, "w") as f:
        json.dump(document_mapping, f)
    logger.info("Initialized new document mapping.")

# =====================
# Initialize or Rebuild FAISS Vector Store
# =====================

if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded existing FAISS vector store.")
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        logger.info("Rebuilding FAISS vector store from document mapping.")
        rebuild_faiss()
else:
    logger.info("No existing FAISS vector store found. Initializing FAISS vector store.")
    rebuild_faiss()


# =====================
# Core Functions
# =====================


def get_gpt_response(user_input):
    """
    Generate a response from GPT-4 based on user input and contextual embeddings.
    """
    try:
        if vector_store is not None:
            # Retrieve relevant context from vector store
            relevant_docs = vector_store.similarity_search(user_input, k=10)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Log the retrieved context
            for idx, doc in enumerate(relevant_docs):
                snippet = (
                    doc.page_content[:100].replace("\n", " ") + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                logger.info(f"Retrieved Doc {idx + 1}: {snippet}")

            # Update memory with the context
            memory.chat_memory.add_user_message(f"Context:\n{context}")
        else:
            context = ""
            logger.warning("Vector store is empty. No context available.")

        # Add user message to memory
        memory.chat_memory.add_user_message(user_input)

        # Get response from LangChain's conversation
        response = conversation.predict(input=user_input)
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
        if vector_store is not None:
            # Retrieve relevant context from vector store
            relevant_docs = vector_store.similarity_search(user_input, k=10)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Log the retrieved context
            for idx, doc in enumerate(relevant_docs):
                snippet = (
                    doc.page_content[:100].replace("\n", " ") + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                logger.info(f"Retrieved Doc {idx + 1}: {snippet}")

            # Update memory with the context
            memory.chat_memory.add_user_message(f"Context:\n{context}")
        else:
            context = ""
            logger.warning("Vector store is empty. No context available.")

        # Add user message to memory
        memory.chat_memory.add_user_message(user_input)

        # Stream the response using ConversationChain's predict method
        # Note: LangChain's ConversationChain does not natively support streaming.
        # Therefore, we'll interact directly with ChatOpenAI for streaming.

        # Initialize ChatOpenAI with streaming
        # stream_llm = ChatOpenAI(
        #    temperature=0.7,
        #    model_name="gpt-4o",
        #    streaming=True  # Enable streaming
        # )

        # Prepare the conversation messages
        messages = memory.chat_memory.messages
        ai_response = ""
        # Start streaming the response
        for chunk in llm.stream(messages):
            # Each chunk is an AIMessageChunk object
            content = chunk.content
            if content:
                ai_response += content
                yield content

        # After streaming, add AI's response to memory
        memory.chat_memory.add_ai_message(ai_response)

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
    if document_id not in document_mapping:
        logger.warning(f"Document ID {document_id} not found in mapping.")
        raise ValueError("Document ID not found.")

    # Retrieve the source from the document metadata
    source = document_mapping[document_id]["metadata"]["source"]

    # Remove documents from document_mapping.json
    to_delete_ids = [doc_id for doc_id, doc in document_mapping.items() if doc["metadata"]["source"] == source]
    for doc_id in to_delete_ids:
        del document_mapping[doc_id]
    logger.info(f"Removed {len(to_delete_ids)} documents associated with source '{source}' from document mapping.")

    # Save the updated document mapping
    with open(mapping_path, "w") as f:
        json.dump(document_mapping, f)

    # Rebuild FAISS vector store to reflect deletions
    logger.info("Rebuilding FAISS vector store to reflect deletions.")
    rebuild_faiss()


# After indexing or deletion, update all_documents
def update_all_documents():
    """
    Fetch all current documents from the record manager.
    """
    global all_documents
    try:
        records = record_manager.get_all_records()
        all_documents = [Document(page_content=record.page_content, metadata=record.metadata) for record in records]
        logger.info(f"Updated all_documents list with {len(all_documents)} documents.")
    except Exception as e:
        logger.error(f"Failed to update all_documents: {e}")
        all_documents = []


@app.route("/")
def home():
    """
    Redirect the root URL to the chat page.
    """
    return redirect(url_for("chat"))


@app.route("/upload_page", methods=["GET"])
def upload_page():
    """
    Render the upload page via a GET request.
    """
    # Get unique sources
    unique_sources = list(set([doc["metadata"]["source"] for doc in document_mapping.values()]))
    return render_template("upload.html", sources=unique_sources)


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle file uploads, extract text, generate embeddings, and update the vector store.
    """

    # Check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        logger.warning("No file part in the request.")
        return redirect(url_for("upload_page"))

    file = request.files["file"]

    # If user does not select file, browser may submit an empty part
    if file.filename == "":
        flash("No selected file")
        logger.warning("No file selected for upload.")
        return redirect(url_for("upload_page"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        flash("File successfully uploaded")
        logger.info(f"File uploaded: {filename}")

        # Extract text from the uploaded file
        try:
            texts = extract_text(file_path, filename)
            logger.info(f"Extracted {len(texts)} text chunks from {filename}.")
        except ValueError as ve:
            flash(str(ve))
            logger.error(f"Error extracting text: {ve}")
            return redirect(url_for("upload_page"))

        if texts:
            try:
                # Initialize FAISS vector store if it's None
                if vector_store is None:
                    logger.info("FAISS vector store is not initialized. Rebuilding FAISS vector store.")
                    rebuild_faiss()
                    if vector_store is None:
                        flash("Failed to initialize the vector store. Please check the logs.")
                        return redirect(url_for("upload_page"))

                # Create Document objects with 'source' metadata
                source_id = filename  # Using filename as the source identifier
                new_documents = [
                    Document(page_content=text.page_content, metadata={"source": source_id}) for text in texts
                ]

                # Add new documents to FAISS vector store
                if vector_store is not None:
                    vector_store.add_documents(new_documents)
                    logger.info(f"Added {len(new_documents)} new documents to FAISS vector store.")

                # Update the local document mapping
                for doc in new_documents:
                    document_id = str(uuid.uuid4())
                    document_mapping[document_id] = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "file_path": file_path,
                    }

                # Save the updated document mapping
                with open(mapping_path, "w") as f:
                    json.dump(document_mapping, f)

                # Save the FAISS vector store to disk
                if vector_store is not None:
                    vector_store.save_local(vector_store_path)
                    logger.info("FAISS vector store saved to disk after upload.")

                # Update all_documents list if needed (Not used anymore)
                # update_all_documents()  # Removed as we rely on document_mapping.json

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
    else:
        flash("Allowed file types are pdf, docx, txt")
        logger.warning(f"Attempted to upload unsupported file type: {file.filename}")
        return redirect(url_for("upload_page"))


@app.route("/chat", methods=["GET"])
def chat():
    """
    Render the chat interface.
    """
    return render_template("chat.html", document_mapping=document_mapping)


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get GPT response
    response = get_gpt_response(user_input)

    return jsonify({"response": response})


@app.route("/stream_response", methods=["POST"])
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
        return jsonify({"message": f"Document {document_id} removed successfully."}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error removing document {document_id}: {e}")
        return jsonify({"error": "Failed to remove document."}), 500


@app.route("/delete_sources", methods=["POST"])
def delete_sources():
    """
    Handle deletion of selected sources.
    Expects form data with 'sources' as a list of source names.
    """
    global vector_store
    selected_sources = request.form.getlist("sources")

    if not selected_sources:
        flash("No sources selected for deletion.")
        logger.warning("No sources selected for deletion.")
        return redirect(url_for("upload_page"))

    try:
        # Fetch all current documents
        all_current_docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in document_mapping.values()
        ]

        # Rebuild FAISS vector store after removing selected sources
        logger.info("Rebuilding FAISS vector store after source deletions.")
        vector_store = None  # Reset FAISS vector store
        with open(mapping_path, "w") as f:
            # Remove documents associated with selected sources
            to_delete_ids = [
                doc_id for doc_id, doc in document_mapping.items() if doc["metadata"]["source"] in selected_sources
            ]
            for doc_id in to_delete_ids:
                del document_mapping[doc_id]
            # Save the updated document mapping
            json.dump(document_mapping, f)

        # Rebuild FAISS vector store
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
