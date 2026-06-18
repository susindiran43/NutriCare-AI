# NutriCare AI — Intelligent Clinical Nutrition Dashboard

NutriCare AI is a high-performance, portfolio-quality clinical decision support system that maps physical symptoms to precise, scientifically-backed dietary recommendations. Built on a hybrid pipeline combining machine learning classification and dense retrieval (RAG) capabilities, it allows clinicians and patients to analyze symptoms, retrieve recovery guidelines, and run semantic queries over healthcare datasets in real-time.

---

## 🌟 Key Features

- **Disease Predictor (Inference Engine)**: Leverages a trained Random Forest Classifier on a structured symptom vocabulary to compute condition likelihoods along with statistical classification confidence scores.
- **RAG-Powered Guidance (FAISS Semantic Retrieval)**: Falls back to dense vector database matching when conditions fall outside exact string matches. Embeddings are calculated using SentenceTransformers (`all-MiniLM-L6-v2`) and searched against a local L2 flat index.
- **Natural Language Search Portal**: Enables users to write conversational queries (e.g., *'recommended foods for gastric pain'*) to run semantic vector similarity scans.
- **Interactive Vitals Dashboard**: Simulates and displays patient vitals dynamically based on condition severity (Confidence, Recovery outlook, Hydration target, Risk Level, and general Nutrition Grade).
- **Responsive 3-Column Glassmorphic UI**: Beautiful, lightweight light-glass interface styled using premium vanilla CSS transitions, custom micro-interactions, accessibility attributes, and dynamic autocomplete lists.
- **Optimized for Low-Memory Deployments**: Replaced heavy pandas run-time memory structures with direct stream CSV processing and lazy-loaded large weights into memory on-demand. Startup footprint is reduced by >70%, preventing deployment failures on platforms like Render or HuggingFace.

---

## 🏗 System Architecture

```mermaid
graph TD
    User([User Symptoms Selection]) -->|JSON Payload| FlaskAPI[Flask REST Controller]
    
    subgraph Backend Core (Memory Optimized)
        FlaskAPI -->|Feature Matrix Vector| MLModel{Random Forest Model}
        MLModel -->|Predicted Disease + Confidence| RetrievalOrchestrator[Retrieval Orchestrator]
        
        RetrievalOrchestrator -->|Exact Match Query| LocalDict{CSV Local Dictionary}
        RetrievalOrchestrator -->|Semantic Fallback| DenseRetriever[FAISS Vector Index]
        
        DenseRetriever -->|Embeddings Query| SentenceTransformer[SentenceTransformer Model]
        SentenceTransformer -->|Index Query| FAISS[FlatL2 Index Search]
    end
    
    LocalDict -->|JSON Response| UI[Light-Glass Frontend UI]
    FAISS -->|Top Match Guidance| UI
    
    UI -->|Render Vitals| Metrics[Health Analytics Grid]
```

---

## 📁 Directory Structure

```text
NutriCare-AI/
├── app.py                     # Main Flask Application REST endpoints & lazy-loading
├── train_model.py             # Random Forest training script to generate models
├── create_vector_db.py        # SentenceTransformers pipeline generating FAISS index
├── download_model.py          # HuggingFace downloader utility for model weights
├── requirements.txt           # Lightened, optimized runtime dependencies
├── runtime.txt                # Python environment lockfile
├── render.yaml                # Render Web Service deployment configuration
├── .env.example               # Environment variables configuration template
├── data/
│   ├── symptoms.csv.csv       # Training dataset for ML classifiers
│   ├── Symptom-severity.csv   # Symptom severity mapping
│   ├── symptom_Description.csv # Condition definitions
│   ├── symptom_precaution.csv # Recommended precautions
│   └── nutrition_knowledge.csv# RAG/FAISS ground truth data
├── models/
│   ├── disease_model.pkl      # Trained classifier binary (Random Forest)
│   └── all-MiniLM-L6-v2/      # Localized HuggingFace embedding weights
├── vector_db/
│   └── faiss_index            # Compiled FAISS index binary
├── static/
│   ├── style.css              # Premium theme styling and CSS keyframe animations
│   └── script.js              # State manager, autocomplete, and UI controllers
└── templates/
    └── index.html             # High-contrast accessibility structured HTML5
```

---

## 🛠 Tech Stack

- **Framework**: Python 3.11, Flask
- **Machine Learning**: Scikit-Learn, NumPy, Pickle
- **Vector Search / RAG**: FAISS (Facebook AI Similarity Search), SentenceTransformers (`all-MiniLM-L6-v2`)
- **Frontend**: Vanilla HTML5 (Semantic Structure, ARIA compliant), CSS3 (Custom Variables, Keyframe Mesh Gradients, Flexbox/Grid), Modern JS (ES6+, LocalStorage session cache)
- **Deployment**: Render Web Services compatibility configuration

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.10+ installed on your system.

### 1. Clone & Initialize Environment
```bash
git clone https://github.com/yourusername/NutriCare-AI.git
cd NutriCare-AI
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Linux/macOS**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Models & Database Indexes (Optional)
If the binaries do not exist or you want to retrain the pipeline:
```bash
# 1. Download transformer weights
python download_model.py

# 2. Train Random Forest Classifier
python train_model.py

# 3. Create FAISS Vector Index
python create_vector_db.py
```

### 4. Run Locally
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## ☁️ Deployment Guide

This repository includes custom configuration scripts for deployment:

### Deploying to Render
1. Create a new **Web Service** on Render.
2. Link your GitHub repository.
3. Render will automatically read the `render.yaml` configuration:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python download_model.py`
   - **Start Command**: `python app.py`
4. Set the `PORT` environment variable if desired. (The application dynamically binds to port configured via `PORT` env).
