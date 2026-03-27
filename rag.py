from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

print("🚀 Local RAG Chatbot Started")

# Load embeddings
embeddings = HuggingFaceEmbeddings()

# Load vector DB
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# Load Local LLM
llm = ChatOllama(
    model="mistral"
)

while True:
    query = input("\nAsk Question (type exit): ")

    if query == "exit":
        break

    docs = retriever.invoke(query)

    context = docs[0].page_content

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