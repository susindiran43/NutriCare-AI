import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

def clean_symptom(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    # Correct key typos in the symptom dataset to match Symptom-severity.csv
    if s == 'dischromic _patches':
        return 'dischromic_patches'
    if s == 'foul_smell_of urine':
        return 'foul_smell_ofurine'
    if s == 'spotting_ urination':
        return 'spotting_urination'
    s = s.replace(' ', '_')
    s = s.replace('__', '_')
    return s

def train():
    print("Loading symptoms dataset...")
    df = pd.read_csv('data/symptoms.csv.csv')
    
    print("Extracting unique symptoms...")
    unique_symptoms = set()
    for col in df.columns[1:]:
        for val in df[col].dropna():
            cleaned = clean_symptom(val)
            if cleaned:
                unique_symptoms.add(cleaned)
    
    symptoms_list = sorted(list(unique_symptoms))
    print(f"Found {len(symptoms_list)} unique symptoms.")
    
    symptom_to_idx = {sym: idx for idx, sym in enumerate(symptoms_list)}
    
    print("Creating feature matrix X and labels y...")
    X = np.zeros((len(df), len(symptoms_list)), dtype=np.float32)
    y = df['Disease'].values
    
    for idx, row in df.iterrows():
        for col in df.columns[1:]:
            val = row[col]
            cleaned = clean_symptom(val)
            if cleaned in symptom_to_idx:
                X[idx, symptom_to_idx[cleaned]] = 1.0
                
    print(f"Feature matrix shape: {X.shape}")
    
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    accuracy = clf.score(X, y)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/disease_model.pkl'
    print(f"Saving model to {model_path}...")
    
    model_data = {
        'model': clf,
        'symptoms': symptoms_list,
        'symptom_to_idx': symptom_to_idx,
        'classes': clf.classes_.tolist()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model training and saving completed successfully!")

if __name__ == '__main__':
    train()
