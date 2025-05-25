import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found! Please set OPENAI_API_KEY in your environment or .env file.")
    st.stop()

# Constants
PDF_DIR = "./Data"
CHROMA_PERSIST_DIR = "./chroma_medical_db"

# Load and split PDF documents
@st.cache_resource(show_spinner=False)
def load_and_split_pdfs(pdf_dir):
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Create or load the Chroma vectorstore
@st.cache_resource(show_spinner=False)
def create_or_load_vectorstore(_documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        st.info("Loaded existing vectorstore.")
    else:
        vectorstore = Chroma.from_documents(_documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        st.success("Created new vectorstore and persisted to disk.")
    return vectorstore

# Streamlit UI
st.title("🩺 Medical Chatbot with RAG")

with st.spinner("📚 Loading and indexing medical documents..."):
    docs = load_and_split_pdfs(PDF_DIR)
    vectorstore = create_or_load_vectorstore(docs)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Input UI
query = st.text_input("Ask me a medical question:")

if query:
    with st.spinner("🤖 Generating answer..."):
        answer = qa.run(query)
    st.markdown("**Answer:**")
    st.write(answer)
