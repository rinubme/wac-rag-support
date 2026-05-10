# WAC RAG - Washington Administrative Code AI Assistant

Local RAG system for answering questions about Washington State regulations.

## Setup

1. Install Ollama → https://ollama.com
2. Download WAC PDFs and put them in `wac_data/` folder
3. Run:
   ```bash
   pip install -r requirements.txt
   ollama run llama3.2
   python app.py