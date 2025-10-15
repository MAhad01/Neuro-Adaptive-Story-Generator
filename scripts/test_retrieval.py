# File: test_retrieval.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration (Should match your other scripts) ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def test_database_retrieval():
    """
    Connects to the ChromaDB and performs a test query to see what it retrieves.
    """
    print("--- ⚙️ Running Retrieval Test ---")
    
    try:
        # 1. Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 2. Connect to the existing vector store
        print(f"Connecting to database at: {CHROMA_DB_PATH}")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        # 3. Define a test query
        query = "Going to school for the first time"
        print(f"\nPerforming search with query: '{query}'")
        
        # 4. Perform the search
        results = vector_store.similarity_search(query, k=2)
        
        # 5. Print the results
        if not results:
            print("\n--- ❌ RESULT: No documents found. The database is likely empty or the content is not relevant. ---")
        else:
            print(f"\n--- ✅ RESULT: Found {len(results)} document(s)! ---")
            for i, doc in enumerate(results):
                print(f"\n--- Document {i+1} ---")
                print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
                print("--------------------")

    except Exception as e:
        print(f"\n--- 🚨 An error occurred ---")
        print(f"Error: {e}")
        print("This might mean the database directory doesn't exist or is corrupted.")

if __name__ == "__main__":
    test_database_retrieval()