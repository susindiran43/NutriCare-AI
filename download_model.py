import os
from sentence_transformers import SentenceTransformer

def download():
    model_path = './models/all-MiniLM-L6-v2'
    print("Checking/Downloading SentenceTransformer model...")
    os.makedirs('./models', exist_ok=True)
    # This downloads the model from Hugging Face hub if not present
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")

if __name__ == '__main__':
    download()
