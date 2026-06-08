import os
import pickle
import pandas as pd
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

app = Flask(__name__)

# Cache variables
model_data = None
description_map = {}
precaution_map = {}
nutrition_df = None
nutrition_norm_map = {}
sentence_model = None
faiss_index = None

def normalize_name(d):
    if not isinstance(d, str):
        return ""
    return d.strip().lower().replace("hemmorhoids", "hemorrhoids").replace("diseae", "disease")

def initialize_app():
    global model_data, description_map, precaution_map, nutrition_df, nutrition_norm_map, sentence_model, faiss_index
    
    print("Initializing components...")
    
    # 1. Load ML Model
    model_path = 'models/disease_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("ML Model loaded successfully.")
    else:
        print(f"Warning: ML Model file not found at {model_path}")

    # 2. Load Descriptions
    desc_path = 'data/symptom_Description.csv'
    if os.path.exists(desc_path):
        df_desc = pd.read_csv(desc_path)
        for _, row in df_desc.iterrows():
            key = normalize_name(row['Disease'])
            description_map[key] = row['Description']
        print(f"Loaded {len(description_map)} descriptions.")
    else:
        print(f"Warning: Descriptions CSV not found at {desc_path}")

    # 3. Load Precautions
    prec_path = 'data/symptom_precaution.csv'
    if os.path.exists(prec_path):
        df_prec = pd.read_csv(prec_path)
        for _, row in df_prec.iterrows():
            key = normalize_name(row['Disease'])
            precautions = []
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                val = row[col]
                if isinstance(val, str) and val.strip():
                    precautions.append(val.strip().capitalize())
            precaution_map[key] = precautions
        print(f"Loaded {len(precaution_map)} precaution lists.")
    else:
        print(f"Warning: Precautions CSV not found at {prec_path}")

    # 4. Load Nutrition Data
    nutr_path = 'data/nutrition_knowledge.csv'
    if os.path.exists(nutr_path):
        nutrition_df = pd.read_csv(nutr_path)
        for idx, row in nutrition_df.iterrows():
            key = normalize_name(row['Disease'])
            nutrition_norm_map[key] = {
                'disease': row['Disease'],
                'recommended': row['Recommended Food'],
                'avoid': row['Avoid Food'],
                'reason': row['Reason']
            }
        print(f"Loaded {len(nutrition_norm_map)} nutrition entries.")
    else:
        print(f"Warning: Nutrition knowledge CSV not found at {nutr_path}")

    # 5. Load Sentence Transformer
    print("Loading SentenceTransformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 6. Load FAISS index
    faiss_path = 'vector_db/faiss_index'
    if os.path.exists(faiss_path):
        faiss_index = faiss.read_index(faiss_path)
        print("FAISS Index loaded successfully.")
    else:
        print(f"Warning: FAISS index not found at {faiss_path}")

# Initialize components before handling requests
initialize_app()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    if model_data is None:
        return jsonify({"error": "ML model data not loaded"}), 500
    # Returns the list of sorted clean symptom strings
    return jsonify({"symptoms": model_data['symptoms']})

@app.route('/api/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({"error": "ML model data not loaded"}), 500
    
    data = request.json or {}
    user_symptoms = data.get('symptoms', [])
    
    if not user_symptoms:
        return jsonify({"error": "No symptoms selected"}), 400
        
    # Build feature vector
    symptoms_list = model_data['symptoms']
    symptom_to_idx = model_data['symptom_to_idx']
    X_input = np.zeros((1, len(symptoms_list)), dtype=np.float32)
    
    matched_count = 0
    for sym in user_symptoms:
        if sym in symptom_to_idx:
            X_input[0, symptom_to_idx[sym]] = 1.0
            matched_count += 1
            
    if matched_count == 0:
        return jsonify({"error": "None of the selected symptoms match the model vocabulary"}), 400
        
    # Classify
    clf = model_data['model']
    prediction = clf.predict(X_input)[0]
    
    # Predict probabilities (if applicable)
    try:
        probs = clf.predict_proba(X_input)[0]
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
    except:
        confidence = 1.0
        
    norm_predicted = normalize_name(prediction)
    
    # Lookup Description & Precautions
    description = description_map.get(norm_predicted, "Description not available for this condition.")
    precautions = precaution_map.get(norm_predicted, ["Consult a doctor for advice."])
    
    # Lookup Nutrition Guidance (Exact match first, then fall back to FAISS vector search)
    diet_info = None
    if norm_predicted in nutrition_norm_map:
        diet_info = nutrition_norm_map[norm_predicted]
        diet_info['match_method'] = 'exact'
    elif faiss_index is not None and sentence_model is not None and nutrition_df is not None:
        # Embed the predicted disease name and query the FAISS database
        q_emb = sentence_model.encode([prediction])
        D, I = faiss_index.search(np.array(q_emb).astype('float32'), k=1)
        match_idx = I[0][0]
        
        if match_idx >= 0 and match_idx < len(nutrition_df):
            row = nutrition_df.iloc[match_idx]
            diet_info = {
                'disease': row['Disease'],
                'recommended': row['Recommended Food'],
                'avoid': row['Avoid Food'],
                'reason': row['Reason'],
                'match_method': 'semantic'
            }
            
    if diet_info is None:
        diet_info = {
            'disease': 'General Wellness',
            'recommended': 'Water, fresh fruits, vegetables, and simple soups',
            'avoid': 'Spicy foods, sugary drinks, deep fried items, and heavy processed meals',
            'reason': 'To support generalized immune response and ensure simple digestion',
            'match_method': 'fallback'
        }
        
    return jsonify({
        "disease": prediction,
        "confidence": confidence,
        "description": description,
        "precautions": precautions,
        "nutrition": diet_info
    })

@app.route('/api/search', methods=['POST'])
def search():
    if faiss_index is None or sentence_model is None or nutrition_df is None:
        return jsonify({"error": "Vector database components not fully loaded"}), 500
        
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "Empty search query"}), 400
        
    # Perform FAISS search
    q_emb = sentence_model.encode([query])
    # Search for top 3 matches
    k = min(3, len(nutrition_df))
    D, I = faiss_index.search(np.array(q_emb).astype('float32'), k=k)
    
    results = []
    for rank, match_idx in enumerate(I[0]):
        if match_idx >= 0 and match_idx < len(nutrition_df):
            row = nutrition_df.iloc[match_idx]
            score = float(D[0][rank])
            # Normalize index scores (distance) into a relative match percentage if desired,
            # or just return raw details
            results.append({
                'disease': row['Disease'],
                'recommended': row['Recommended Food'],
                'avoid': row['Avoid Food'],
                'reason': row['Reason'],
                'distance': score
            })
            
    return jsonify({"results": results})

@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    if nutrition_df is None:
        return jsonify({"error": "Nutrition data not loaded"}), 500
    
    catalog_list = []
    for _, row in nutrition_df.iterrows():
        catalog_list.append({
            'disease': row['Disease'],
            'recommended': row['Recommended Food'],
            'avoid': row['Avoid Food'],
            'reason': row['Reason']
        })
        
    # Sort alphabetically by disease name
    catalog_list.sort(key=lambda x: x['disease'])
    return jsonify({"catalog": catalog_list})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
