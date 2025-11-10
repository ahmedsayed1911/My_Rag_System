import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# ----- Document Chain -----
def create_stuff_documents_chain(llm, prompt):
    class SimpleDocumentChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt

        def combine_docs(self, docs, input=None):
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt_text = self.prompt.format(context=context, input=input)
            response = self.llm.invoke(prompt_text)
            return response.content if hasattr(response, "content") else str(response)

    return SimpleDocumentChain(llm, prompt)


# ----- Retrieval Chain -----
def create_retrieval_chain(retriever, vectorstore, document_chain):
    class SimpleRetrievalChain:
        def __init__(self, retriever, vectorstore, document_chain):
            self.retriever = retriever
            self.vectorstore = vectorstore
            self.document_chain = document_chain

        def invoke(self, inputs):
            query = inputs.get("input", "")
            retrieved_docs = None

            # جرب كل الطرق الممكنة لأي نسخة من Chroma
            try:
                if hasattr(self.retriever, "get_relevant_documents"):
                    retrieved_docs = self.retriever.get_relevant_documents(query)
                elif hasattr(self.retriever, "similarity_search"):
                    retrieved_docs = self.retriever.similarity_search(query, k=4)
            except Exception:
                pass

            # fallback لو retriever فشل
            if not retrieved_docs:
                if hasattr(self.vectorstore, "similarity_search"):
                    retrieved_docs = self.vectorstore.similarity_search(query, k=4)
                else:
                    raise AttributeError("No valid search method found in Chroma retriever.")

            answer = self.document_chain.combine_docs(retrieved_docs, input=query)
            return {"answer": answer}

    return SimpleRetrievalChain(retriever, vectorstore, document_chain)


# ----- Streamlit Setup -----
st.set_page_config(page_title="My RAG System")

st.title("My RAG System - Ask your PDF")

api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        # حفظ الملف مؤقتًا لقراءته
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.write(f"Loaded {len(docs)} pages from {uploaded_file.name}")

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
        retrieval_chain = create_retrieval_chain(retriever, vectorstore, document_chain)

        st.success("PDF processed successfully.")

        question = st.text_input("Ask a question about your PDF:")
        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": question})
                st.write("Answer:")
                st.write(response["answer"])

else:
    st.info("Please upload a PDF file to begin.")
