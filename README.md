---

### **How to Run the Project**

1.  **Activate Environment**: Open a command prompt in your project folder and run: `.\venv\Scripts\activate`
2.  **Prepare Data**: Run the data preparation script: `python scripts/01_prepare_data.py`
3.  **Ingest Data**: Run the ingestion script to build your vector database: `python scripts/02_ingest_to_chromadb.py`
4.  **Fine-Tune Model**: Open and run all cells in the Jupyter Notebook: `notebooks/03_finetune_llm.ipynb`. This will take some time and will create the `models/gpt2-social-story-finetuned` folder.
5.  **Run the App**: Launch the Streamlit application: `streamlit run app.py`

Your web browser should open with the application running.

---

### **Ethical Considerations**

As highlighted by the research, developing AI tools for this purpose requires a strong ethical framework[cite: 69].

* **Human-in-the-Loop**: This system is designed to **assist**, not replace, caregivers, clinicians, or parents[cite: 75]. The generated story must **always** be reviewed and approved by a responsible adult before being shared with the individual[cite: 76].
* **Data Privacy**: The current setup runs entirely on your local machine. No data is sent to external services, ensuring privacy.
* **Bias and Fairness**: The model's output is heavily influenced by its training data. The provided dataset is very small. A real-world application would require a much larger, carefully curated, and diverse dataset co-designed with autistic individuals to avoid promoting masking or neurotypical-centric solutions[cite: 73, 74].
* **Dignity and Autonomy**: The goal is to empower the individual, not enforce conformity[cite: 77]. The stories should be seen as a tool to help understand and navigate situations, reducing anxiety and promoting autonomy.