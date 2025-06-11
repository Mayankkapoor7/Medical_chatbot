import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical RAG Chatbot")

PDF_DIR = "./Data"
CHROMA_PERSIST_DIR = "./chroma_medical_db"

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found!")
    st.stop()
@st.cache_resource(show_spinner=False)
def load_and_split_pdfs(pdf_dir):
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
@st.cache_resource(show_spinner=False)
def create_or_load_vectorstore(_documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        return Chroma.from_documents(_documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)

schemas = [
    ResponseSchema(name="reasoning", description="Step-by-step reasoning or medical inference"),
    ResponseSchema(name="diagnosis", description="Possible condition or outcome"),
    ResponseSchema(name="advice", description="Recommended next steps or actions"),
    ResponseSchema(name="disclaimer", description="Medical disclaimer and caution")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()
custom_prompt = PromptTemplate(
    template="""
You are Dr. MedAI, a trusted AI medical assistant.
Use the following context to respond to the user.
Only use medical facts from the context and your knowledge.
Structure your response as instructed below.

Context:
{context}

Question:
{question}

{format_instructions}
""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions}
)
with st.spinner("📚 Loading and indexing documents..."):
    try:
        docs = load_and_split_pdfs(PDF_DIR)
        vectorstore = create_or_load_vectorstore(docs)
        st.success(f"Loaded {len(docs)} chunks.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        st.stop()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)
query = st.text_input("💬 Ask a medical question:")
if query:
    with st.spinner("🔍 Generating structured response..."):
        try:
            result = qa_chain({"query": query})
            parsed = parser.parse(result["result"])

            st.markdown("Medical Advice")
            st.markdown(f"**Reasoning:** {parsed['reasoning']}")
            st.markdown(f"**Diagnosis:** {parsed['diagnosis']}")
            st.markdown(f"**Advice:** {parsed['advice']}")
            st.markdown(f"**Disclaimer:** {parsed['disclaimer']}")

            with st.expander("Context and Prompt"):
                context_text = "\n\n".join([doc.page_content[:300] for doc in result["source_documents"]])
                prompt_view = custom_prompt.format(context=context_text, question=query)
                st.code(prompt_view, language="markdown")

        except Exception as e:
            st.error(f"Error during answer generation: {e}")