# Setup the vector database for RAG using Ollama embeddings and ChromaDB.
# This update creates different vector databases depending on the type of data.

import os
import sys
import glob
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

# --- Configuration (Based on our earlier discussion) ---
# This directory will hold the vector database files (so you don't re-index every time)
VECTOR_DB_PATH = "./ollama_cv_example/chroma_db_career" 
DATA_DIRECTORY = "/home/pete/code/agentic_demo/ollama_cv_example/data" 

# Map file extensions to the appropriate LangChain Loader class
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".pdf": PyPDFLoader,
}

# Embedding model MUST match the full tag you pulled from Ollama
EMBEDDING_MODEL = 'qllama/bge-small-en-v1.5:latest' 

def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Loads all supported documents from a directory using the appropriate loader."""
    all_documents = []
    
    # Iterate through every file in the data directory
    for filepath in glob.glob(os.path.join(directory_path, "**/*"), recursive=True):
        if os.path.isfile(filepath):
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            if ext in LOADER_MAPPING:
                Loader = LOADER_MAPPING[ext]
                try:
                    loader = Loader(filepath)
                    all_documents.extend(loader.load())
                    print(f"Loaded: {filepath}")
                except Exception as e:
                    print(f"Error loading {filepath} with {Loader.__name__}: {e}")
            # Optional: Ignore files without a mapped loader silently, 
            # or add an 'else' block to print a warning.
            
    if not all_documents:
        print(f"Error: No supported documents found in the '{directory_path}' directory.")
        sys.exit(1)
        
    return all_documents

def get_retriever():
    """
    Loads documents, indexes them into ChromaDB, or loads the existing index.
    Returns a LangChain Retriever object.
    """
    # Initialize the Ollama Embeddings object
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print("✅ Loading existing vector store...")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print(f"🛠️ Indexing documents from {DATA_DIRECTORY} for the first time...")
        
        # 1. Load Documents
        # TextLoader is used here because your document is a text file
        #loader = TextLoader(DOCUMENT_FILE)
        data = load_documents_from_directory(DATA_DIRECTORY)

        # 2. Split Documents
        # RecursiveCharacterTextSplitter splits documents intelligently
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(data)

        # 3. Create Vector Store and persist to disk
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("✨ Indexing complete and saved to disk.")

    # 4. Create a Retriever
    # This component searches the vector store for the top 5 relevant document chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

if __name__ == '__main__':
    # Run the setup script if executed directly
    get_retriever()
    print("\nSetup script finished. You can now run rag_chat.py.")