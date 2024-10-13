# app.py

import os
import openai
from openai import RateLimitError

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
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

from langchain.indexes import SQLRecordManager, index
import faiss

from cachetools import TTLCache, cached

import json
import numpy as np
import uuid  # Ensure uuid is imported


# =====================
# Logging Configuration
# =====================
import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',  # Define the log message format
    handlers=[
        logging.StreamHandler()  # Log messages will be output to the console
    ]
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
app.secret_key = os.getenv('SECRET_KEY')

# Configure upload settings
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# =====================
# Caching Configuration
# =====================

# Initialize embedding cache with TTL of 1 hour and max size of 1000
cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour TTL

# =====================
# Helper Functions
# =====================


# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == 'txt':
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
    model_name="gpt-4o"
)
conversation = ConversationChain(llm=llm, memory=memory)
logger.info(f"Using OpenAI model: {llm.model_name}")


# Define paths for vector store and document mapping
vector_store_path = 'data/vector_store.faiss'
mapping_path = 'data/document_mapping.json'
record_manager_path = 'data/record_manager_cache.sql'

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
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    max_retries=6  # Increase the number of retries
)

@cached(cache)
def get_cached_embedding(text):
    return embeddings.embed_query(text)



# Create or load the FAISS vector store
vector_store = None
all_documents = []
if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        all_documents = vector_store.vectorstore.get_all_documents()
        logger.info("Loaded existing FAISS vector store.")
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        logger.info("FAISS vector store will be initialized upon first document upload.")
else:
    logger.info("No existing FAISS vector store found. It will be created upon first upload.")


# Build a list of all current documents from the vector store
if vector_store is not None:
    all_documents = vector_store.vectorstore.get_all_documents()
else:
    all_documents = []

# Load or initialize the document mapping
if os.path.exists(mapping_path):
    try:
        with open(mapping_path, 'r') as f:
            # Check if the file is empty
            if os.path.getsize(mapping_path) > 0:
                document_mapping = json.load(f)
                logger.info("Loaded existing document mapping.")
            else:
                # File is empty, initialize empty dict
                document_mapping = {}
                with open(mapping_path, 'w') as fw:
                    json.dump(document_mapping, fw)
                logger.warning("document_mapping.json was empty. Initialized as empty dictionary.")
    except json.JSONDecodeError:
        # JSON is malformed, initialize empty dict
        document_mapping = {}
        with open(mapping_path, 'w') as fw:
            json.dump(document_mapping, fw)
        logger.error("document_mapping.json was malformed. Reinitialized as empty dictionary.")
else:
    document_mapping = {}
    with open(mapping_path, 'w') as f:
        json.dump(document_mapping, f)
    logger.info("Initialized new document mapping.")


def initialize_faiss():
    """
    Initialize the FAISS vector store upon the first document upload.
    """
    global vector_store
    try:
        # Determine the embedding dimension from a sample embedding
        sample_embedding = embeddings.embed_query("test")
        embedding_dim = len(sample_embedding)
        logger.info(f"Embedding dimension determined: {embedding_dim}")
        
        # Create a FAISS index
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=faiss_index,
            docstore= InMemoryDocstore(),
            index_to_docstore_id={}
            )
        logger.info("FAISS vector store initialized with IndexFlatL2.")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS vector store: {e}")
        vector_store = None


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
                snippet = doc.page_content[:100].replace('\n', ' ') + '...' if len(doc.page_content) > 100 else doc.page_content
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

def remove_document(document_id):
    """
    Remove a document and its embeddings from the FAISS vector store and document mapping.
    """
    if document_id not in document_mapping:
        logger.warning(f"Document ID {document_id} not found in mapping.")
        raise ValueError("Document ID not found.")
    
    # Retrieve the source from the document metadata
    source = document_mapping[document_id]['metadata']['source']
    
    # Fetch all documents associated with this source
    source_docs = [
        Document(
            page_content=doc['page_content'],
            metadata=doc['metadata']
        )
        for doc in document_mapping.values()
        if doc['metadata']['source'] == source
    ]
    
    # Perform full cleanup for this source
    # To delete all documents from this source, pass an empty list for these documents
    # However, the indexing API expects a full list of all documents to retain
    # Therefore, retrieve all documents excluding those from the source to be deleted
    all_current_docs = [
        Document(
            page_content=doc['page_content'],
            metadata=doc['metadata']
        )
        for doc in document_mapping.values()
    ]
    
    # Exclude documents from the source to be deleted
    updated_docs = [
        doc for doc in all_current_docs
        if doc.metadata['source'] != source
    ]
    
    # Perform full cleanup with the updated list
    indexing_result = index(
        updated_docs,
        record_manager,
        vector_store,
        cleanup="full",
        source_id_key="source"
    )
    
    logger.info(f"Indexing result during deletion: {indexing_result}")
    
    # Remove all documents associated with this source from the local mapping
    to_delete = [doc_id for doc_id, doc in document_mapping.items() if doc['metadata']['source'] == source]
    for doc_id in to_delete:
        del document_mapping[doc_id]
    
    # Save the updated document mapping
    with open(mapping_path, 'w') as f:
        json.dump(document_mapping, f)
    
    # Save the FAISS vector store to disk
    if vector_store is not None:
        vector_store.save_local(vector_store_path)
        logger.info("FAISS vector store saved to disk after deletion.")

# After indexing or deletion, update all_documents
def update_all_documents():
    """
    Fetch all current documents from the record manager.
    """
    global all_documents
    try:
        records = record_manager.get_all_records()
        all_documents = [
            Document(
                page_content=record.page_content,
                metadata=record.metadata
            )
            for record in records
        ]
        logger.info(f"Updated all_documents list with {len(all_documents)} documents.")
    except Exception as e:
        logger.error(f"Failed to update all_documents: {e}")
        all_documents = []

@app.route('/')
def home():
    """
    Redirect the root URL to the chat page.
    """
    return redirect(url_for('chat'))

@app.route('/upload_page', methods=['GET'])
def upload_page():
    """
    Render the upload page via a GET request.
    """
    return render_template('upload.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads, extract text, generate embeddings, and update the vector store.
    """
    global vector_store  # Declare as global to modify the global variable
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        logger.warning("No file part in the request.")
        redirect(url_for('upload_page'))
    
    file = request.files['file']
    
    # If user does not select file, browser may submit an empty part
    if file.filename == '':
        flash('No selected file')
        logger.warning("No file selected for upload.")
        redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File successfully uploaded')
        logger.info(f"File uploaded: {filename}")
        
        # Extract text from the uploaded file
        try:
            texts = extract_text(file_path, filename)
            logger.info(f"Extracted {len(texts)} text chunks from {filename}.")
        except ValueError as ve:
            flash(str(ve))
            logger.error(f"Error extracting text: {ve}")
            redirect(url_for('upload_page'))
        
        if texts:
            try:
                # Initialize FAISS vector store if it's None
                if vector_store is None:
                    initialize_faiss()
                    if vector_store is None:
                        flash("Failed to initialize the vector store. Please check the logs.")
                        redirect(url_for('upload_page'))
                
                # Create Document objects with 'source' metadata
                source_id = filename  # Using filename as the source identifier
                new_documents = [
                    Document(page_content=text.page_content, metadata={"source": source_id})
                    for text in texts
                ]
                
                # Index the new documents with 'incremental' cleanup
                indexing_result = index(
                    new_documents,
                    record_manager,
                    vector_store,
                    cleanup="incremental",
                    source_id_key="source"
                )
                
                logger.info(f"Indexing result: {indexing_result}")
                
                # Update the local document mapping
                for doc in new_documents:
                    document_id = str(uuid.uuid4())
                    document_mapping[document_id] = {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata,
                        'file_path': file_path
                    }
                
                # Save the updated document mapping
                with open(mapping_path, 'w') as f:
                    json.dump(document_mapping, f)

                # Save the FAISS vector store to disk
                if vector_store is not None:
                    vector_store.save_local(vector_store_path)
                    logger.info("FAISS vector store saved to disk after upload.")
                
                
                flash('File content integrated into context')
            except RateLimitError as e:
                flash("Rate limit exceeded while processing your file. Please try again later.")
                logger.error(f"Rate limit exceeded during embedding: {e}")
            except Exception as e:
                flash("An error occurred while processing your file. Please try again.")
                logger.error(f"Error during embedding or saving vector store: {e}")
        else:
            flash('No text extracted from the uploaded file')
            logger.warning("No text extracted from the uploaded file.")
       
        return redirect(url_for('chat'))
    else:
        flash('Allowed file types are pdf, docx, txt')
        logger.warning(f"Attempted to upload unsupported file type: {file.filename}")
        return redirect(url_for('upload_page'))

@app.route('/chat', methods=['GET'])
def chat():
    """
    Render the chat interface.
    """
    return render_template('chat.html', document_mapping=document_mapping)

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Get GPT response
    response = get_gpt_response(user_input)
    
    return jsonify({'response': response})

@app.route('/remove_document', methods=['POST'])
def remove_document_route():
    """
    Endpoint to remove a document and its embeddings from the context.
    Expects JSON data with 'document_id'.
    """
    data = request.get_json()
    document_id = data.get('document_id')
    
    if not document_id:
        return jsonify({'error': 'No document_id provided'}), 400
    
    try:
        remove_document(document_id)
        flash(f'Document {document_id} removed successfully.')
        return jsonify({'message': f'Document {document_id} removed successfully.'}), 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except Exception as e:
        logger.error(f"Error removing document {document_id}: {e}")
        return jsonify({'error': 'Failed to remove document.'}), 500


# =====================
# Error Handlers
# =====================

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 Internal Server Errors.
    """
    logger.error(f"Server Error: {error}")
    return render_template('500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 Not Found Errors.
    """
    logger.error(f"Not Found Error: {error}")
    return render_template('404.html'), 404

# =====================
# Run the Application
# =====================

if __name__ == '__main__':
    app.run(debug=True)
