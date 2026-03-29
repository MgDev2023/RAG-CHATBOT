import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("📄 RAG AI Chatbot")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    return retriever, llm

try:
    retriever, llm = load_system()
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

query = st.text_input("Ask Question")

if query:
    docs = retriever.invoke(query)

    if not docs:
        st.warning("No relevant documents found for your question.")
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke([HumanMessage(content=prompt)])

        st.write("### ✅ AI Answer")
        st.write(response.content)