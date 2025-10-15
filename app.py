# File: app.py

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FINETUNED_MODEL_PATH = "models/gpt2-social-story-finetuned"

# --- Caching to speed up model loading ---
@st.cache_resource
def load_models():
    """Loads the retriever and the fine-tuned generator model."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)
    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=200,
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
    preferences = st.text_input("Likes/Preferences (optional):", "Loves dinosaurs and the color blue")
    triggers = st.text_input("Dislikes/Triggers (optional):", "Dislikes loud noises")
    generate_button = st.button("Generate Story", type="primary")

# --- Story Generation ---
st.header("Generated Social Story")

if generate_button:
    if not situation:
        st.warning("Please describe the social situation.")
    else:
        with st.spinner("Generating your personalized story..."):
            # 1. Retrieve context
            retrieved_docs = retriever.get_relevant_documents(situation)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            st.subheader("Retrieved Context:")
            st.info(context if context else "No relevant context was found. The model will rely on its training.")

            # 2. Create a simpler, more direct prompt
            prompt_template = f"""<|startoftext|>This is a social story for a child named {child_name} about {situation}.

Here is the story:
My name is {child_name}. {context}"""

            # 3. Generate the story
            raw_output = generator(prompt_template)[0]['generated_text']
            # Clean up the output to only show the generated story
            story = raw_output.replace(prompt_template, "").replace("<|endoftext|>", "").strip()

            st.success(story)
else:
    st.info("Fill out the details in the sidebar and click 'Generate Story'.")