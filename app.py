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
os.makedirs(DATA_DIR, exist_ok=True)

# Local models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3.2", temperature=0.3, num_ctx=8192)

# ===================== VECTOR STORE =====================
def initialize_vectorstore():
    try:
        if not os.path.exists(CHROMA_DIR) or (os.path.exists(DATA_DIR) and len([f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]) > 0):
            print("Loading and indexing WAC documents...")
            loader = PyPDFDirectoryLoader(DATA_DIR)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = splitter.split_documents(docs)
            
            if splits:
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )
                print(f"Successfully indexed {len(splits)} chunks.")
            else:
                print("No PDF files found in wac_data folder.")
        else:
            print("Loading existing vector database...")
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        return vectorstore

    except Exception as e:
        print(f"Vectorstore Error: {e}")
        raise

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        print(f"Question received: {question[:100]}...")

        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}" 
            for d in docs
        ])

        prompt = f"""You are a helpful Washington State WAC assistant.
Answer ONLY based on the context below. Cite WAC numbers when possible.

Context:
{context}

Question: {question}
Answer:"""

        response = llm.invoke(prompt)

        sources = [d.metadata.get('source', 'Unknown') for d in docs]

        return jsonify({
            "answer": response.strip(),
            "sources": sources
        })

    except Exception as e:
        print(f"ERROR processing question: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("WAC RAG Customer Support System Starting...")
    print("Make sure Ollama is running with command: ollama run llama3.2")
    print("Put your WAC PDF files in the wac_data folder")
    app.run(debug=True, port=5000)