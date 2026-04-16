# RAG Chatbot

A chatbot that answers questions based on your own documents, not just general knowledge.

---

## What does it do?

RAG stands for **Retrieval-Augmented Generation**. Instead of the AI making things up, it:
1. Searches through your uploaded documents
2. Finds the most relevant parts
3. Uses those to answer your question

This means the answers are based on your actual data.

---

## How it works

- Documents are converted into a vector database using **FAISS**
- When you ask a question, it finds the closest matching text chunks
- The matched text is sent to **Llama 3.1** (via Groq API) to generate the final answer
- The UI is built with Streamlit

---

## Tech used

- Python
- LangChain (RAG pipeline)
- FAISS (vector database)
- HuggingFace Embeddings (sentence-transformers)
- Groq API with Llama 3.1
- Streamlit (web app)

---

## How to run it locally

```bash
git clone https://github.com/MgDev2023/RAG-CHATBOT.git
cd RAG-CHATBOT
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

Then run:
```bash
streamlit run app.py
```

---

## Files

```
RAG-CHATBOT/
├── app.py           ← Streamlit web UI
├── rag.py           ← command-line version
├── requirements.txt
├── Data/            ← put your documents here
└── vector_db/       ← auto-generated vector store
```

---

## Made by

Megan — portfolio project to learn RAG, LangChain, and LLM integration.
