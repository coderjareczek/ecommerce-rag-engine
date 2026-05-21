from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import load_and_process_reviews, create_vector_store
from rag_chain import get_rag_response
import os

app = FastAPI(title="E-commerce Review Insight Engine")

app.state.vector_store = None

@app.on_event("startup")
def startup_event():
    if os.path.exists("data/reviews.csv"):
        docs = load_and_process_reviews("data/reviews.csv")
        app.state.vector_store = create_vector_store(docs)
        print("Application stated successfully.")
    else:
        print("Warning: Data file not found! Endpoints may fail.")
    
class QueryRequest(BaseModel):
    product_id: str
    query: str

@app.post("/feature-query")
def query_reviews(request: QueryRequest):
    if not app.state.vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")
    
    try:
        answer = get_rag_response(
            vector_store=app.state.vector_store,
            product_id=request.product_id,
            user_query=request.query
        )
        return {"product_id": request.product_id, "query": request.query, "analysis": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product-summary/{product_id}")
def get_product_summary(product_id: str):
    if not app.state.vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")
    
    summary_query = "Generate a concise summary in bullet points: Main Advantages and Main Disadvantages of this product based on the reviews."
    
    try:
        answer = get_rag_response(app.state.vector_store, product_id, summary_query)
        return {"product_id": product_id, "summary": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))