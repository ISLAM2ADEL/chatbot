# pdf_processor.py
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, PDF_PATH

def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_chunks(text, chunk_size=500, chunk_overlap=100):
    nltk.download("punkt_tab")
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap + " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(chunks, embedding_model)

def setup_retriever():
    pdf_text = load_pdf_text(PDF_PATH)
    chunks = create_chunks(pdf_text)
    db = create_vector_store(chunks)
    return db.as_retriever(search_kwargs={"k": 3})
