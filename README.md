# 🛒 E-commerce Review Insight Engine

A 100% local Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, and Ollama (Llama 3) to analyze e-commerce product reviews without external API costs.

## Quickstart

**1. Install dependencies:**
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
**2. Start the local LLM:**
Make sure you have [Ollama](https://ollama.com) installed, then run:

```bash
ollama run llama3
```
**3. Run the application:**

```bash
uvicorn main:app --reload
```
## API Endpoints
Once the server is running, visit http://127.0.0.1:8000/docs to use the interactive Swagger UI.

* `GET /product-summary/{product_id}` - Generates a bulleted list of main advantages and disadvantages.

* `POST /feature-query` - Ask specific questions about a product (e.g., "How is the build quality?").
