# File: scripts/01b_prepare_finetuning_data.py

import os
import re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# --- Configuration ---
CORPUS_PATH = "./corpus_documents"  # Path to your folder of PDFs
OUTPUT_FILE = "./data/finetuning_dataset.txt"
MIN_STORY_LENGTH = 50 # Minimum characters to be considered a valid story

def clean_text(text):
    """
    Cleans up the text extracted from PDFs.
    - Removes excessive newlines and whitespace.
    - Removes potential headers/footers with URLs or page numbers.
    """
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Remove page numbers (e.g., "Page 5", "5 / 10")
    text = re.sub(r'(Page\s\d+|\d+\s\/\s\d+)', '', text)
    # Remove common website URLs
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove any text that is likely a copyright or footer
    text = re.sub(r'©.*', '', text)
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    return text

def create_finetuning_data_from_corpus():
    """
    Loads all PDF documents, cleans their content, and formats them
    into a single text file for fine-tuning.
    """
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: The directory '{CORPUS_PATH}' was not found.")
        return

    print(f"--- Creating fine-tuning dataset from PDFs in '{CORPUS_PATH}' ---")

    # 1. Load Documents
    loader = DirectoryLoader(
        CORPUS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()

    if not documents:
        print("No PDF documents found. Aborting.")
        return

    print(f"Loaded content from {len(documents)} documents.")

    # 2. Process and format each document
    formatted_stories = []
    for doc in documents:
        # Each PDF is treated as a potential source of one or more stories.
        # For simplicity, we treat each document as one story.
        cleaned_content = clean_text(doc.page_content)

        if len(cleaned_content) > MIN_STORY_LENGTH:
            # Format with start and end tokens for the model
            formatted_story = f"<|startoftext|>{cleaned_content}<|endoftext|>"
            formatted_stories.append(formatted_story)

    # 3. Save to the output file
    print(f"Saving {len(formatted_stories)} formatted stories to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for story in formatted_stories:
            f.write(story + '\n')

    print("--- Fine-tuning dataset created successfully! ---")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
    create_finetuning_data_from_corpus()