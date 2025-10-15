# File: scripts/02_ingest_to_chromadb.py

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader # <-- Import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
CORPUS_PATH = "./corpus_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def ingest_data():
    """
    Loads PDF and TXT documents from a directory, splits them into chunks,
    creates embeddings, and stores them in a persistent ChromaDB vector store.
    """
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: The directory '{CORPUS_PATH}' was not found.")
        return

    print("--- Starting data ingestion from PDF and TXT documents ---")

    # --- CHANGE 1: Load PDFs and TXT files separately ---
    print(f"Loading PDF documents from {CORPUS_PATH}...")
    pdf_loader = DirectoryLoader(
        CORPUS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    pdf_documents = pdf_loader.load()
    print(f"Loaded {len(pdf_documents)} PDF documents.")

    print(f"Loading TXT documents from {CORPUS_PATH}...")
    text_loader = DirectoryLoader(
        CORPUS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader, # <-- Use the correct loader for text files
        show_progress=True
    )
    text_documents = text_loader.load()
    print(f"Loaded {len(text_documents)} TXT documents.")
    
    # Combine the lists of documents
    documents = pdf_documents + text_documents
    print(f"Total documents to process: {len(documents)}")

    if not documents:
        print("No documents found in the directory. Aborting.")
        return

    # 2. Split into Chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")

    # 3. Embed and Store
    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating and persisting ChromaDB at {CHROMA_DB_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print("--- ✅ Data ingestion complete! ---")
    print(f"Vector store has been updated with {len(documents)} documents.")

if __name__ == "__main__":
    ingest_data()