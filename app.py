# app.py

import os
import openai
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
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
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Initialize LangChain components globally
memory = ConversationBufferMemory()
llm = OpenAI(temperature=0.7)
conversation = ConversationChain(llm=llm, memory=memory)

# Initialize vector store for context from uploaded files
embeddings = OpenAIEmbeddings()
vector_store_path = 'vector_store.faiss'

if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(vector_store_path, embeddings)
else:
    vector_store = None  # Initialize as None when no documents are present

def get_gpt_response(user_input):
    if vector_store is not None:
        # Retrieve relevant context from vector store
        relevant_docs = vector_store.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = ""
    
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
    global vector_store  # Declare as global to modify the global variable
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
        
        if texts:
            if vector_store is None:
                # Initialize FAISS with the first set of documents
                vector_store = FAISS.from_documents(texts, embeddings)
                flash('Vector store created and file content integrated into context')
            else:
                # Add new documents to the existing vector store
                vector_store.add_documents(texts)
                flash('File content integrated into context')
            
            # Save the updated vector store
            vector_store.save_local(vector_store_path)
        else:
            flash('No text extracted from the uploaded file')
        
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are pdf, docx, txt')
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

if __name__ == '__main__':
    app.run(debug=True)
