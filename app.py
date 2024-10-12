# app.py

import os
import openai
from openai.error import RateLimitError
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from cachetools import TTLCache, cached


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

@cached(cache)
def get_cached_embedding(text):
    return embeddings.embed_query(text)

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

# Initialize vector store for context from uploaded files
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Specify the embedding model
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    max_retries=6  # Increase the number of retries
#    retry_delay=4  # Adjust the initial delay
)
vector_store_path = 'vector_store.faiss'

if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        logger.info("Loaded existing FAISS vector store.")
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        vector_store = None
else:
    vector_store = None  # Initialize as None when no documents are present
    logger.info("No existing FAISS vector store found. It will be created upon first upload.")

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
            relevant_docs = vector_store.similarity_search(user_input, k=3)
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



@app.route('/')
def index():
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
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser may submit an empty part
    if file.filename == '':
        flash('No selected file')
        logger.warning("No file selected for upload.")
        return redirect(request.url)
    
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
            return redirect(request.url)
        
        if texts:
            try:
                # Generate embeddings using cached function
                for text in texts:
                    get_cached_embedding(text.page_content)
                logger.info("Embeddings generated and cached.")
                
                if vector_store is None:
                    # Initialize FAISS with the first set of documents
                    vector_store = FAISS.from_documents(texts, embeddings)
                    logger.info("Initialized FAISS vector store with uploaded documents.")
                    flash('Vector store created and file content integrated into context')
                else:
                    # Add new documents to the existing vector store
                    vector_store.add_documents(texts)
                    logger.info(f"Added {len(texts)} new documents to FAISS vector store.")
                    flash('File content integrated into context')
                
                # Save the updated vector store
                vector_store.save_local(vector_store_path)
                logger.info("FAISS vector store saved successfully.")
            except RateLimitError as e:
                flash("Rate limit exceeded while processing your file. Please try again later.")
                logger.error(f"Rate limit exceeded during embedding: {e}")
            except Exception as e:
                flash("An error occurred while processing your file. Please try again.")
                logger.error(f"Error during embedding or saving vector store: {e}")
        else:
            flash('No text extracted from the uploaded file')
            logger.warning("No text extracted from the uploaded file.")
        
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are pdf, docx, txt')
        logger.warning(f"Attempted to upload unsupported file type: {file.filename}")
        return redirect(request.url)


@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Get GPT response
    response = get_gpt_response(user_input)
    
    return jsonify({'response': response})


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
