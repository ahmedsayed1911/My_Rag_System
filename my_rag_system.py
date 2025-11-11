import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# إعداد الصفحة
st.set_page_config(page_title="My RAG System")
st.title("My RAG System - Ask your PDF")

# مفتاح الـ API
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# رفع ملف PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        # حفظ الملف مؤقتًا لقراءته
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # تحميل النصوص من الملف
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")

        # تقسيم النصوص
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # إنشاء الـ Embeddings وقاعدة البيانات
        embedding = FastEmbedEmbeddings()
        vectorstore = Chroma.from_documents(splits, embedding=embedding)
        retriever = vectorstore.as_retriever()

        # إعداد الموديل
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=api_key,
            temperature=0.5,
            max_tokens=512
        )

        # إعداد الـ Prompt
        prompt = ChatPromptTemplate.from_template("""
        Use the following context to answer the question.
        If you don't know the answer, just say "I don't know."

        Context:
        {context}

        Question: {input}
        """)

        # بناء الـ Document Chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # بناء الـ Retrieval Chain
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

        st.success("PDF processed successfully.")

        # السؤال من المستخدم
        question = st.text_input("Ask a question about your PDF:")
        if st.button("Get Answer") and question:
            with st.spinner("Generating answer..."):
                response = retrieval_chain.invoke({"input": question})
                st.write("Answer:")
                st.write(response["answer"])

else:
    st.info("Please upload a PDF file to begin.")
