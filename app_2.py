import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Attention RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Attention Paper - RAG Chatbot")
st.write("Ask questions about the Attention Is All You Need paper.")


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

loader = PyPDFLoader("Attention.pdf")
documents = loader.load()


# --------------------------------------------------
# Text Splitting
# --------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000
)

chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# --------------------------------------------------
# Embeddings + ChromaDB
# --------------------------------------------------

#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Create the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_rag_minilm",
    persist_directory="./chroma_minilm_db"
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

from langchain_groq import ChatGroq

model_groq = ChatGroq(
    model="qwen/qwen3.8-27b"
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# --------------------------------------------------
# Create Document Chain
# --------------------------------------------------

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

document_chain = create_stuff_documents_chain(
    model_groq,
    prompt
)


# --------------------------------------------------
# Create Retrieval Chain
# --------------------------------------------------

from langchain_classic.chains import create_retrieval_chain

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about Attention Is All You Need..."
)


if question:

    # Display question
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the paper..."):

            response = rag_chain.invoke({
                "input": question
            })

        st.write(response["answer"])