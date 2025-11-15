import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A complete vector database wrapper using ChromaDB + HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):

        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Persistent ChromaDB storage
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    # CHUNKING
    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        return splitter.split_text(text)

    # ADD DOCUMENTS
    def add_documents(self, documents: List[str]) -> None:
        print(f"Processing {len(documents)} documents...")

        ids = []
        chunks = []
        embeddings = []

        for doc_index, doc_text in enumerate(documents):
            doc_chunks = self.chunk_text(doc_text)

            for chunk_index, chunk in enumerate(doc_chunks):
                chunk_id = f"doc{doc_index}_chunk{chunk_index}"

                ids.append(chunk_id)
                chunks.append(chunk)

        if len(chunks) == 0:
            print("⚠ No chunks generated. Check document content.")
            return

        # Generate embeddings for all chunks
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Store in ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )

        print(f"✅ Added {len(chunks)} chunks to vector database")

    # SEARCH
    def search(self, query: str, n_results: int = 5) -> List[str]:

        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        # If collection is empty or no match
        if not results.get("documents") or len(results["documents"][0]) == 0:
            return []

        # Return only document strings (chunk texts)
        return results["documents"][0]