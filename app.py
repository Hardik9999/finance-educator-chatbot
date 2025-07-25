import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# === SETTINGS ===
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3-70b-8192"

# === INIT LLM ===
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name=MODEL_NAME
    )

# === BUILD / LOAD VECTOR DB ===
@st.cache_resource
def load_or_create_db():
    if os.path.exists(DB_DIR):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        loader = DirectoryLoader(DATA_DIR, glob='*.pdf', loader_cls=PyPDFLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory=DB_DIR)
        vector_db.persist()
        return vector_db

# === SETUP RETRIEVAL CHAIN ===
def setup_chain(llm, vector_db):
    retriever = vector_db.as_retriever()
    prompt = PromptTemplate(
        template="""
You are a helpful Finance & Investment Educator chatbot. Answer questions clearly using only factual data from financial and tax documents. Avoid giving personal advice.

Context:
{context}

User: {question}
Assistant:""",
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# === MAIN APP ===
st.title("💰 Finance & Investment Educator")
st.caption("Ask anything about personal finance, investing, or Indian tax policies.")

user_input = st.text_input("Your Question")

if user_input:
    llm = initialize_llm()
    db = load_or_create_db()
    qa_chain = setup_chain(llm, db)

    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": user_input})
        st.success(result["result"])
