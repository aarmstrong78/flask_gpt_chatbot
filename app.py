# app.py

import os
import openai
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain import OpenAI, ConversationChain, VectorStore
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Configure upload settings
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

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
        loader = Docx2txtLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Initialize LangChain components
def initialize_conversation():
    memory = ConversationBufferMemory()
    llm = OpenAI(temperature=0.7)
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation

# Initialize vector store for context from uploaded files
def initialize_vector_store():
    embeddings = OpenAIEmbeddings()
    # Check if vector store already exists
    if os.path.exists('vector_store.faiss'):
        vector_store = FAISS.load_local('vector_store.faiss', embeddings)
    else:
        vector_store = FAISS.from_documents([], embeddings)
    return vector_store


def get_gpt_response(user_input, conversation, vector_store):
    # Retrieve relevant context from vector store
    relevant_docs = vector_store.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Combine user input with context
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nAI:"
    
    # Get response from LangChain's conversation
    response = conversation.predict(input=user_input)
    return response


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If user does not select file, browser may submit an empty part
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File successfully uploaded')
        
        # Extract text from the uploaded file
        try:
            texts = extract_text(file_path, filename)
        except ValueError as ve:
            flash(str(ve))
            return redirect(request.url)
        
        # Initialize or load vector store
        vector_store = initialize_vector_store()
        
        # Add new documents to the vector store
        vector_store.add_documents(texts)
        
        # Save the updated vector store
        vector_store.save_local('vector_store.faiss')
        
        flash('File content integrated into context')
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are pdf, docx, txt')
        return redirect(request.url)

@app.route('/chat', methods=['GET'])
def chat():
    if 'conversation' not in session:
        session['conversation'] = initialize_conversation()
    if 'vector_store' not in session:
        session['vector_store'] = initialize_vector_store()
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Initialize conversation and vector store from session
    conversation = session.get('conversation')
    vector_store = session.get('vector_store')
    
    if not conversation or not vector_store:
        conversation = initialize_conversation()
        vector_store = initialize_vector_store()
        session['conversation'] = conversation
        session['vector_store'] = vector_store
    
    # Get GPT response
    response = get_gpt_response(user_input, conversation, vector_store)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
