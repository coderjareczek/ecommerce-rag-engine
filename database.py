import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_process_reviews(file_path: str):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['review_text'])

    documents = []
    for _, row in df.iterrows():
        metadata = {
            "product_id": str(row['product_id']),
            "rating": int(row['rating'])
        }
        doc = Document(page_content=row['review_text'], metadata=metadata)
        documents.append(doc)

    text_splttter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return text_splttter.split_documents(documents)

def create_vector_store(docs):
    print("Initializing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store