import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

# --- Streamlit UI ---
st.set_page_config(page_title="My RAG System", page_icon="🤖")
st.title("📚 My RAG System — Ask Your PDF")

# Get API Key
api_key = st.secrets["GROQ_API_KEY"]
if not api_key:
    st.error("Please set your GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(splits, embedding=embedding)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=512
    )

    prompt = PromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, just say "I don't know."

    Context:
    {context}

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("✅ PDF processed successfully!")

    # Ask Question
    question = st.text_input("💬 Ask a question about your PDF:")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": question})
            st.write("### 🤖 Answer:")
            st.write(response["answer"])
