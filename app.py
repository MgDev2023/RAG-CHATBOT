import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

st.title("📄 RAG AI Chatbot")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    llm = ChatOllama(model="mistral")

    return retriever, llm

retriever, llm = load_system()

query = st.text_input("Ask Question")

if query:
    docs = retriever.invoke(query)

    context = docs[0].page_content

    prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.write("### ✅ AI Answer")
    st.write(response.content)