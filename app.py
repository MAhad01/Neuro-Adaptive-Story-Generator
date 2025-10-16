# File: app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# --- NEW: Update the model path to the new distilgpt2 model ---
FINETUNED_MODEL_PATH = "models/distilgpt2-social-story-finetuned"

# --- Caching to speed up model loading ---
@st.cache_resource
def load_models():
    """Loads the retriever and the fine-tuned generator model."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # This will now load the tokenizer with the special tokens correctly defined
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)
    
    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id
    )
    return retriever, generator

# --- Main Application Logic ---
st.set_page_config(layout="wide")
st.title("🤖 Personalized Social Story Generator")
st.markdown("This tool helps create social stories for autistic individuals, leveraging AI to tailor the narrative to specific situations and user needs.")

retriever, generator = load_models()
st.success("Models loaded successfully!")

# --- User Input Form ---
with st.sidebar:
    st.header("👤 Personalization")
    child_name = st.text_input("Child's Name:", "Alex")
    child_age = st.slider("Child's Age:", 3, 18, 6)
    st.header(" situasi")
    situation = st.text_area("Describe the social situation:", "Going to a new school for the first time.")
    preferences = st.text_input("Likes/Preferences (optional):", "Loves dinosaurs")
    triggers = st.text_input("Dislikes/Triggers (optional):", "Dislikes loud noises")
    generate_button = st.button("Generate Story", type="primary")

# --- Story Generation ---
st.header("Generated Social Story")

if generate_button:
    if not situation:
        st.warning("Please describe the social situation.")
    else:
        with st.spinner("Generating your personalized story..."):
            retrieved_docs = retriever.get_relevant_documents(situation)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            st.subheader("Retrieved Context:")
            st.info(context if context else "No relevant context was found. The model will rely on its training.")

            # A cleaner, more direct prompt for better results
            prompt_template = f"<|startoftext|>Write a simple social story for {child_name}, who is {child_age} years old, about {situation}. The story should be reassuring. Start the story with 'I am going to...' or a similar phrase.\n\nHere is the story:"

            raw_output = generator(prompt_template)[0]['generated_text']
            story = raw_output.split("Here is the story:")[-1].strip()

            st.success(story)
else:
    st.info("Fill out the details in the sidebar and click 'Generate Story'.")