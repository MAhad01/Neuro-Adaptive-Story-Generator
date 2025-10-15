# File: scripts/01_prepare_data.py

import pandas as pd

def create_rag_corpus():
    """Creates a CSV file of social stories for the RAG knowledge base."""
    stories_data = {
        'title': [
            "Going to the Grocery Store",
            "My First Day at School",
            "Visiting the Dentist",
            "Sharing Toys with Friends",
            "What to do When I Feel Angry"
        ],
        'content': [
            "Sometimes, we need to go to the grocery store to buy food. The store can be bright and have many people. I will stay close to my caregiver. We will get a cart and put our food inside. If I feel overwhelmed, I can take a deep breath. At the end, we pay for the food and go home.",
            "I am going to a new school. I will meet my teacher and new friends. I will have my own desk and a place for my backpack. We will learn new things like reading and math. If I feel shy, it's okay. I can tell my teacher if I need help. School is a safe place to learn and play.",
            "I will go to the dentist to keep my teeth healthy. A dentist is a doctor for teeth. I will sit in a big, comfy chair. The dentist will use small tools to look at my teeth. It is important to stay still. If I am scared, I can hold a toy or squeeze my hands. The dentist helps my smile stay bright.",
            "When my friends come to play, we can share toys. Sharing means we take turns. I can let my friend play with my car, and then it will be my turn. If I want a toy my friend is using, I can ask, 'Can I have a turn, please?'. Sharing makes playtime fun for everyone.",
            "Sometimes I might feel angry. My face might feel hot and my fists might clench. It's okay to feel angry, but it is not okay to hit or yell. When I feel angry, I can go to a quiet place. I can take three deep breaths. I can also punch a pillow or tell an adult how I am feeling. This helps the angry feeling go away safely."
        ]
    }
    df = pd.DataFrame(stories_data)
    df.to_csv('data/social_stories_corpus.csv', index=False)
    print("Successfully created data/social_stories_corpus.csv for RAG.")

def create_finetuning_data():
    """Creates a simple text file for fine-tuning the LLM."""
    # This dataset format is simple for demonstration. Each story is a line.
    # In a real project, you'd want thousands of high-quality examples.
    stories = [
        "<|startoftext|>Going to the library means I have to be quiet. I will whisper if I need to talk. I will find a book and sit down to read. It is a peaceful place.<|endoftext|>",
        "<|startoftext|>When I meet someone new, I can say 'hello'. I can try to smile. I don't have to talk a lot. Just saying hello is friendly.<|endoftext|>",
        "<|startoftext|>It's time for a haircut. I will sit in a big chair. The barber will use scissors and a comb. The sound of the clippers might be loud, but it doesn't hurt. I will look so nice after.<|endoftext|>",
        "<|startoftext|>When I lose a game, I might feel sad or frustrated. It is okay to lose. I can say 'good game' to the winner. We can play again another time.<|endoftext|>",
        "<|startoftext|>Fire drills at school are for practice. A loud bell will ring. I will line up with my class and walk outside quietly. My teacher will be there. It helps us stay safe.<|endoftext|>"
    ]
    with open('data/finetuning_dataset.txt', 'w', encoding='utf-8') as f:
        for story in stories:
            f.write(story + '\n')
    print("Successfully created data/finetuning_dataset.txt for fine-tuning.")

if __name__ == '__main__':
    # Create the data directory if it doesn't exist
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
        
    create_rag_corpus()
    create_finetuning_data()