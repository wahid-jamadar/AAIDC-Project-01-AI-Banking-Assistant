# **RAG-Based AI-Banking Assistant (AAIDC Project 1)**

🚀 *A fully implemented Retrieval-Augmented Generation system using ChromaDB, HuggingFace embeddings, and multiple LLM providers.*

---

## 🤖 **What is this?**

This project is a fully working **RAG (Retrieval-Augmented Generation)** AI assistant. It allows you to ask questions based on your own documents by combining:

* 📄 Document loading
* 🔍 Vector similarity search
* 🧠 Embedding-based retrieval
* 💬 LLM powered answer generation

Think of it as:

> **"ChatGPT that knows your documents."**

---

## 🎯 **What This Project Can Do**

Your RAG assistant can:

* 📂 Load `.txt` documents from the `data/` directory
* ✂️ Automatically chunk documents
* 🔢 Generate embeddings using SentenceTransformers
* 🧬 Store vectors in a persistent ChromaDB database
* 🔍 Retrieve relevant chunks using similarity search
* 💬 Generate accurate answers using OpenAI, Groq, or Google Gemini
* 🔧 Automatically choose the available LLM provider

---

## 🧠 **How It Works**

1. **Document Loading:** Reads `.txt` files from `data/`.
2. **Chunking:** Splits large documents using LangChain’s `RecursiveCharacterTextSplitter`.
3. **Embeddings:** Converts text into vectors using `all-MiniLM-L6-v2`.
4. **Vector DB:** Stores embeddings in ChromaDB with persistent local storage.
5. **Similarity Search:** Retrieves top-k relevant chunks.
6. **Prompting:** A custom RAG prompt passes context + question to the LLM.
7. **LLM Answer:** OpenAI → Groq → Gemini (auto fallback).

---

## 🧩 **Features**

✔ Fully implemented RAG pipeline <br>
✔ Multi-provider LLM support <br>
✔ ChromaDB persistent storage <br>
✔ Automatic document chunking <br>
✔ Uses HuggingFace embeddings <br>
✔ Clean modular structure <br>
✔ No manual setup inside code <br>
✔ Works with any `.txt` data 

---

## 📦 **Project Structure**

```
Module_01_Project_01/
├── src/
│   ├── app.py            # Main RAG engine
│   └── vectordb.py       # Chroma + Embeddings + Chunking + Search
├── data/
│   ├── banking_data01.txt
│   ├── banking_data02.txt
├── chroma_db/            # Auto-created persistent DB folder
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ **Setup Instructions**

### **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **2️⃣ Configure Your API Key**

Add your API key to `.env`:

```
OPENAI_API_KEY=your_key_here
# OR
GROQ_API_KEY=your_key_here
# OR
GOOGLE_API_KEY=your_key_here
```

> The system will automatically select the first available one.

---

### **3️⃣ Add Your Documents**

Place `.txt` files into the `data/` folder.

Example:

```
data/
├── banking_data01.txt
├── banking_data02.txt
```

---

### **4️⃣ Run the RAG Assistant**

```bash
python src/app.py
```

Example interaction:

```
Your question: What is KYC in banking?

ANSWER:
KYC (Know Your Customer) is a process...
```

---

## 🧪 **Testing the Components**

### **Test Chunking**

```python
from src.vectordb import VectorDB
v = VectorDB()
print(v.chunk_text("Sample document test"))
```

### **Test Vector Search**

```python
v.search("banking")
```

### **Test Full RAG**

Run:

```bash
python src/app.py
```

Ask:

```
Explain the NEFT payment system.
```

---

## 🛠️ **Tech Stack**

* **Python**
* **ChromaDB** (Vector database)
* **SentenceTransformers** (Embeddings)
* **LangChain** (Chunking + Prompting)
* **LLM Providers:**

  * OpenAI
  * Groq (Used In My Project)
  * Google Gemini

---

## 🧑‍💻 **Author**

**Wahid Jamdar** <br>
B.Tech CSE <br>
DY Patil Agriculture & Technical University, Kolhapur <br>

---

## 📄 **License**

This project is created as part of **AAIDC Project-1** and intended for educational use.

---
