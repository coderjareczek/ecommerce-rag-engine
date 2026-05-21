from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_response(vector_store, product_id: str, user_query: str):
    # Retrieve only documents matching the requested product
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"product_id": product_id}, "k": 4}
    )

    llm = ChatOllama(model="llama3", temperature=0)

    system_prompt = (
        "You are a professional e-commerce analytical system.\n"
        "Based on the following excerpts of customer reviews, "
        "answer the user's query accurately. Do not invent facts.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]