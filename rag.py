import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

print("🚀 RAG Chatbot Started (Groq)")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load vector DB
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# Load Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

while True:
    query = input("\nAsk Question (type exit): ")

    if query.strip().lower() == "exit":
        break

    docs = retriever.invoke(query)

    if not docs:
        print("\nNo relevant documents found for your question.")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer based only on the context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\n✅ AI Answer:\n")
    print(response.content)
    print("\n--------------------")