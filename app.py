from flask import Flask, render_template, request, jsonify
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ===================== CONFIG =====================
DATA_DIR = "wac_data"
CHROMA_DIR = "chroma_wac_db"
PORT = int(os.environ.get("PORT", 8080))   # Important for Fly.io

os.makedirs(DATA_DIR, exist_ok=True)

# Use lighter settings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(
    model="phi3",           # Much lighter than llama3.2
    temperature=0.3,
    num_ctx=4096
)

# ===================== VECTOR STORE =====================
def initialize_vectorstore():
    try:
        print("Loading vector database...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}" for d in docs])

        prompt = f"""You are a helpful WAC assistant. Answer concisely using only the context.

Context:
{context}

Question: {question}
Answer:"""

        response = llm.invoke(prompt)

        return jsonify({
            "answer": response.strip(),
            "sources": [d.metadata.get('source', 'Unknown') for d in docs]
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Sorry, service is temporarily unavailable."}), 500

if __name__ == '__main__':
    print(f"WAC RAG starting on port {PORT}")
    print("Using model: phi3")
    app.run(host="0.0.0.0", port=PORT, debug=False)