import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq

# ------------------ LOAD ENV ------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ PAGE ------------------
st.set_page_config(page_title="Document Intelligence Chatbot", layout="wide")

st.markdown("## 📄 Multi-Document Chatbot")
st.caption("Interact with multiple PDF documents using Retrieval-Augmented Generation (RAG) powered by Groq")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# ------------------ SIDEBAR ------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# ------------------ FUNCTIONS ------------------

def extract_text(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


@st.cache_resource(show_spinner=False)
def create_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # smaller = faster
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db


def create_rag_chain(db):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    prompt = PromptTemplate(
        template="""
        Answer ONLY from the context.
        If not available, say: "Answer not in context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})  # faster retrieval

    rag_chain = (
        RunnableMap({
            "context": lambda q: "\n\n".join(
                doc.page_content for doc in retriever.invoke(q)
            ),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    return rag_chain

# ------------------ SESSION STATE ------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ------------------ PROCESS PDFs ------------------

if uploaded_files and st.session_state.rag_chain is None:
    with st.spinner("🔄 Processing documents... Please wait"):
        text = extract_text(uploaded_files)
        db = create_vector_db(text)
        st.session_state.rag_chain = create_rag_chain(db)

    st.success("✅ Documents processed successfully. You can start asking questions.")

# ------------------ CHAT UI ------------------

if st.session_state.rag_chain:

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        with st.spinner("Generating answer..."):
            response = st.session_state.rag_chain.invoke(user_input)
            answer = response.content

        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").markdown(answer)

else:
    st.info("📌 Upload PDF documents to begin")