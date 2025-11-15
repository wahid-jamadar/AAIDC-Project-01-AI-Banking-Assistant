import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load sample documents from the ./data folder.
    Supports: .txt files

    Returns:
        List of text documents
    """
    data_dir = "data"
    results = []

    if not os.path.exists(data_dir):
        print("Data directory not found. Create a 'data' folder and add .txt files.")
        return results

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                results.append(f.read())

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    """

    def __init__(self):
        """Initialize the RAG assistant."""

        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize Vector DB (Chroma / FAISS inside your custom class)
        self.vector_db = VectorDB()

        # RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent RAG assistant.

Use the provided context to answer the question.
If the context does not contain the answer, say: 
"I could not find this in the knowledge base, but here is what I know:" 
and then answer using your general knowledge.

----------------------
CONTEXT:
{context}
----------------------

QUESTION:
{question}

Provide the best possible answer.
""")

        # Final chain combining prompt → LLM → output parser
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """Choose available LLM based on .env file"""

        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Gemini: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        return None

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to vector DB"""
        self.vector_db.add_documents(documents)

    def invoke(self, question: str, n_results: int = 3) -> str:
        """
        RAG query pipeline:
        1. Search vector DB → retrieve chunks
        2. Combine chunks into context
        3. Generate final LLM answer
        """
        # Retrieve chunks
        search_results = self.vector_db.search(question, n_results=n_results)

        # Combine into a single context string
        context = "\n\n".join(search_results) if search_results else "No relevant context found."

        # Run through chain
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })

        return answer


def main():
    """Demo"""

    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} documents")

        assistant.add_documents(sample_docs)

        print("\nAsk anything! Type 'quit' to exit.")
        while True:
            question = input("Your question: ")
            if question.lower() == "quit":
                break

            response = assistant.invoke(question)
            print("\nANSWER:\n", response, "\n")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env with at least one API key.")


if __name__ == "__main__":
    main()