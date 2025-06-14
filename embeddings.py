# embeddings.py
from langchain.chains import RetrievalQA
from llm_setup import setup_llm
from pdf_processor import setup_retriever

def setup_qa_chain():
    llm = setup_llm()
    retriever = setup_retriever()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
