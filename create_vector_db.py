import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

def create_db():
    print("Loading nutritional knowledge dataset...")
    csv_path = 'data/nutrition_knowledge.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    print("Formatting knowledge documents...")
    sentences = []
    for idx, row in df.iterrows():
        s = f"Disease: {row['Disease']}. Recommended: {row['Recommended Food']}. Avoid: {row['Avoid Food']}. Reason: {row['Reason']}"
        sentences.append(s)
        
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating dense vector embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"Generated embeddings of shape: {embeddings.shape}")
    
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs('vector_db', exist_ok=True)
    index_path = 'vector_db/faiss_index'
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print("Vector database creation completed successfully!")

if __name__ == '__main__':
    create_db()
