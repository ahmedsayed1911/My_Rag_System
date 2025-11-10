import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

# --- Handle LangChain version differences safely ---
def create_stuff_documents_chain(llm, prompt):
    """
    Custom simplified version of LangChain's create_stuff_documents_chain,
    compatible with older LangChain versions.
    """
    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.base import BaseCombineDocumentsChain

    class SimpleCombineDocumentsChain(BaseCombineDocumentsChain):
        def _combine_docs(self, docs, **kwargs):
            # دمج النصوص من المستندات في سياق واحد
            context = "\n\n".join([doc.page_content for doc in docs])
            # تمرير السياق إلى النموذج عبر prompt
            return LLMChain(llm=llm, prompt=prompt).run(context=context, **kwargs)

    return SimpleCombineDocumentsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))

# --- Streamlit UI setup ---
st.set_page_config(page_title="My RAG System", page_icon="🤖")
st.title("📚 My RAG System — Ask Your PDF")

# --- API Key ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# --- Upload PDF file ---
uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        # تحميل الـ PDF وتقسيمه
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # تحويل النصوص إلى embeddings وتخزينها في Chroma
        embedding = FastEmbedEmbeddings()
        vectorstore = Chroma.from_documents(splits, embedding=embedding)
        retriever = vectorstore.as_retriever()

        # إعداد نموذج Groq LLaMA 3.3
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=api_key,
            temperature=0.5,
            max_tokens=512
        )

        # إعداد الـ Prompt
        prompt = PromptTemplate.from_template("""
        Use the following context to answer the question.
        If you don't know the answer, just say "I don't know."

        Context:
        {context}

        Question: {input}
        """)

        # إنشاء السلاسل (Chains)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.success("✅ PDF processed successfully!")

        # واجهة السؤال والجواب
        question = st.text_input("💬 Ask a question about your PDF:")
        if st.button("Get Answer") and question:
            with st.spinner("🤔 Thinking..."):
                response = retrieval_chain.invoke({"input": question})
                st.write("### 🤖 Answer:")
                st.write(response["answer"])

                # عرض المصدر (إن وجد)
                sources = response.get("context", None)
                if sources:
                    st.write("### 📚 Sources used:")
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

else:
    st.info("⬆️ Please upload a PDF file to begin.")
