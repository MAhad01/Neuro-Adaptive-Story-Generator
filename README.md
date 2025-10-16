Neuro-Adaptive Story Generator

The Neuro-Adaptive Story Generator is an AI-powered tool designed to create personalized social stories for autistic individuals, particularly children. It leverages a Retrieval-Augmented Generation (RAG) system and a fine-tuned language model to produce narratives that are tailored to a user's specific needs, preferences, and potential anxiety triggers for various social situations.

This project was developed as a proof-of-concept to explore how modern AI techniques can automate and scale the creation of therapeutic tools, making them more accessible to parents, educators, and clinicians.

🚀 Core Features

Personalized Content: Generates stories based on the user's name, age, likes, and dislikes.

Context-Aware Generation: Uses a RAG pipeline with a ChromaDB vector store to retrieve relevant story examples from a corpus of documents.

Adaptive Language Model: Employs a fine-tuned distilgpt2 model that has been trained to write in the specific, reassuring style of a social story.

Interactive Web Interface: Built with Streamlit for an easy-to-use experience.

Ethical by Design: Operates entirely locally, ensuring user data privacy, and is intended as a tool to assist caregivers (human-in-the-loop), not replace them.

🛠️ System Architecture

The project uses a hybrid AI approach to ensure both relevance and stylistic correctness:

Data Corpus: A collection of PDF and text documents containing high-quality social stories forms the knowledge base.

Ingestion Pipeline: The corpus documents are loaded, split into manageable chunks, and converted into vector embeddings using sentence-transformers.

Vector Database: These embeddings are stored in a local ChromaDB instance, allowing for efficient semantic search.

Fine-Tuned LLM: A distilgpt2 model is fine-tuned on a curated dataset of clean, well-structured social stories. This teaches the model the specific format and tone required.

RAG Workflow:

A user query is received through the Streamlit UI.

The retriever searches the ChromaDB for relevant document chunks.

The retrieved context and user personalization details are formatted into a detailed prompt.

The prompt is sent to the fine-tuned model, which generates the final, personalized story.

🔧 Setup and Installation

Follow these steps to get the project running on your local machine.

Prerequisites

Python 3.9+

Windows operating system (though adaptable to others)

Step 1: Clone the Repository & Set Up Environment

# Clone this repository to your local machine (replace with your actual GitHub URL)
git clone [https://github.com/your-username/Neuro-Adaptive-Story-Generator.git](https://github.com/your-username/Neuro-Adaptive-Story-Generator.git)
cd Neuro-Adaptive-Story-Generator

# Create a Python virtual environment
python -m venv SocialStoriesAI

# Activate the virtual environment
.\SocialStoriesAI\Scripts\activate


Step 2: Install Dependencies

Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt


Step 3: Prepare the Data

Add Corpus Documents: Place your collection of social story PDFs and text files into the corpus_documents/ folder.

Create Fine-Tuning Data: Run the script to process your corpus and create a clean dataset for training.

python scripts/01b_prepare_finetuning_data.py


(Optional but Recommended) Manually Curate: For best results, manually review and clean the generated data/finetuning_dataset.txt to ensure it contains only high-quality, complete stories.

Step 4: Build the Vector Database

Run the ingestion script to process your corpus and build the ChromaDB vector store.

# Make sure to delete any existing 'chroma_db' folder first for a clean build
python scripts/02_ingest_to_chromadb.py


Step 5: Fine-Tune the Language Model

Note: The fine-tuned model is not included in this repository due to its large size. You must generate it by running the following notebook.

Run the Jupyter Notebook to fine-tune the distilgpt2 model on your curated dataset. This process will take some time and will save the final model to the models/ directory (which is ignored by Git).

Open and run all cells in notebooks/03_finetune_llm.ipynb.

▶️ How to Run the Application

Once the setup is complete, you can launch the Streamlit web application.

streamlit run app.py


Your web browser will automatically open with the application running. Fill in the details in the sidebar and click "Generate Story" to create a new social story.

💡 Future Improvements

Upgrade to a More Powerful LLM: Experiment with newer, more capable small models like those from the Gemma or Llama families.

Enhance the Corpus: Continuously add more high-quality, diverse social stories to the corpus_documents folder.

UI/UX Improvements: Add features like saving, editing, or exporting the generated stories as PDFs.
