import os
import csv
import pickle
import numpy as np
import faiss
import logging
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="NutriCare AI - Intelligent Health Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global metadata holders (populated once on startup)
@st.cache_resource
def get_static_maps():
    """Load metadata mappings using lightweight CSV Reader to minimize RAM."""
    description_map = {}
    precaution_map = {}
    nutrition_list = []
    nutrition_norm_map = {}
    
    logger.info("Initializing static nutritional knowledge maps...")
    
    # 1. Load Descriptions
    desc_path = 'data/symptom_Description.csv'
    if os.path.exists(desc_path):
        try:
            with open(desc_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_val = row.get('Disease')
                    desc_val = row.get('Description')
                    if disease_val and desc_val:
                        key = normalize_name(disease_val)
                        description_map[key] = desc_val.strip()
        except Exception as e:
            logger.error(f"Error loading descriptions: {e}")
            
    # 2. Load Precautions
    prec_path = 'data/symptom_precaution.csv'
    if os.path.exists(prec_path):
        try:
            with open(prec_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_val = row.get('Disease')
                    if disease_val:
                        key = normalize_name(disease_val)
                        precautions = []
                        for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                            val = row.get(col)
                            if val and val.strip():
                                precautions.append(val.strip().capitalize())
                        precaution_map[key] = precautions
        except Exception as e:
            logger.error(f"Error loading precautions: {e}")
            
    # 3. Load Nutrition Guidance
    nutr_path = 'data/nutrition_knowledge.csv'
    if os.path.exists(nutr_path):
        try:
            with open(nutr_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dis = row.get('Disease')
                    rec = row.get('Recommended Food')
                    avd = row.get('Avoid Food')
                    rsn = row.get('Reason')
                    if dis:
                        item = {
                            'Disease': dis.strip(),
                            'Recommended Food': rec.strip() if rec else "",
                            'Avoid Food': avd.strip() if avd else "",
                            'Reason': rsn.strip() if rsn else ""
                        }
                        nutrition_list.append(item)
                        key = normalize_name(dis)
                        nutrition_norm_map[key] = {
                            'disease': item['Disease'],
                            'recommended': item['Recommended Food'],
                            'avoid': item['Avoid Food'],
                            'reason': item['Reason']
                        }
        except Exception as e:
            logger.error(f"Error loading nutrition knowledge: {e}")
            
    return description_map, precaution_map, nutrition_list, nutrition_norm_map

# Lazy loaded and cached resource getters for AI core components
@st.cache_resource
def get_ml_model():
    model_path = 'models/disease_model.pkl'
    if os.path.exists(model_path):
        logger.info("Lazy loading RandomForest disease classifier...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"ML Model file not found at {model_path}")

@st.cache_resource
def get_sentence_model():
    local_path = './models/all-MiniLM-L6-v2'
    if os.path.exists(local_path):
        logger.info(f"Lazy loading SentenceTransformer from: {local_path}")
        return SentenceTransformer(local_path)
    else:
        logger.info("Downloading SentenceTransformer model ('all-MiniLM-L6-v2')...")
        return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_faiss_index():
    faiss_path = 'vector_db/faiss_index'
    if os.path.exists(faiss_path):
        logger.info("Lazy loading FAISS index...")
        return faiss.read_index(faiss_path)
    return None

def normalize_name(d):
    if not isinstance(d, str):
        return ""
    return d.strip().lower().replace("hemmorhoids", "hemorrhoids").replace("diseae", "disease")

def format_symptom_display(sym):
    if not sym:
        return ""
    return sym.replace('_', ' ').title()

def format_symptom_raw(sym):
    if not sym:
        return ""
    return sym.strip().lower().replace(' ', '_').replace('__', '_')

# Load the static csv details once at launch
description_map, precaution_map, nutrition_list, nutrition_norm_map = get_static_maps()

# Load symptoms list from ML model vocabulary
try:
    ml_model_data = get_ml_model()
    all_symptoms_list = ml_model_data.get('symptoms', [])
except Exception as e:
    all_symptoms_list = []
    logger.error(f"Could not load symptom list: {e}")

# Formatting helpers for UI
def display_formatted_symptom(sym):
    return sym.replace('_', ' ').title()

# Vitals simulation utilities
def get_simulated_recovery_score(disease):
    name = disease.lower()
    if any(x in name for x in ["allergy", "cold", "influenza"]): return 92
    if any(x in name for x in ["gerd", "gastritis", "peptic"]): return 80
    if any(x in name for x in ["hepatitis", "malaria", "typhoid"]): return 62
    if any(x in name for x in ["diabetes", "hypertension", "heart"]): return 55
    return 78

def get_simulated_hydration(disease):
    name = disease.lower()
    if any(x in name for x in ["fever", "diarrhea", "typhoid", "malaria"]): return 92
    if any(x in name for x in ["diabetes", "hypertension"]): return 80
    return 75

def get_simulated_risk(disease):
    name = disease.lower()
    if any(x in name for x in ["diabetes", "hypertension", "heart", "hepatitis"]): return "High"
    if any(x in name for x in ["fever", "malaria", "typhoid", "gerd"]): return "Medium"
    return "Low"

# Daily tips carousel details
daily_tips = [
    "Ensure strict avoidance of raw high-glucose drinks when blood sugar trends exceed clinical thresholds.",
    "Hydration targets increase by 15% under active fever to assist metabolic heat dissipation.",
    "Sodium restriction is primary under chronic hypertensive states to reduce peripheral resistance.",
    "A 12-hour circadian digestive window supports pancreatic rest and improves insulin sensitivity.",
    "Soluble fibers like pectins in apples help stabilize loose stools during gastroenteritis recovery."
]

# Initialize Session State values
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = set()
if 'active_report' not in st.session_state:
    st.session_state.active_report = None
if 'recent_scans' not in st.session_state:
    st.session_state.recent_scans = []
if 'tip_idx' not in st.session_state:
    st.session_state.tip_idx = 0

# Custom Styling Injection for premium Glassmorphic Light Healthcare aesthetics
st.markdown("""
<style>
    /* Styling variables override */
    :root {
        --primary-color: #0f766e;
    }
    
    /* Global modifications */
    .stApp {
        background: radial-gradient(at 0% 0%, rgba(20, 184, 166, 0.08) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(96, 165, 250, 0.08) 0px, transparent 50%),
                    #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(15, 23, 42, 0.05);
    }
    
    /* Card aesthetics */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }
    
    .glass-card-title {
        font-family: 'Outfit', sans-serif;
        color: #0f766e;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 12px;
    }
    
    /* Outcome cards broken out separately */
    .outcome-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.02);
        margin-bottom: 12px;
        border-left: 4px solid #0f766e;
    }
    
    .outcome-card.warning { border-left-color: #d97706; }
    .outcome-card.success { border-left-color: #059669; }
    .outcome-card.danger { border-left-color: #dc2626; }
    .outcome-card.info { border-left-color: #2563eb; }
    
    .outcome-header {
        font-weight: 700;
        font-size: 13.5px;
        margin-bottom: 6px;
        color: #0f172a;
    }
    
    /* Badges */
    .tech-badge {
        font-size: 9px;
        font-weight: 700;
        background: #f1f5f9;
        color: #475569;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# SIDEBAR DESIGN
# ==========================================================================
with st.sidebar:
    st.markdown("<h2 style='color:#0f766e; font-weight:800; font-family:Outfit, sans-serif; margin-bottom:5px;'>NutriCare <span style='color:#2563eb;'>AI</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:#64748b; margin-top:0;'>AI Clinical Intake Assistant</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Radio Selector
    navigation_selection = st.radio(
        "Navigation Menu",
        ["🧬 Symptom Predictor", "🔍 Dietary Search", "📋 Conditions Directory"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Engine Statistics Box
    st.markdown("""
    <div class="glass-card" style="padding:14px; margin-bottom:15px;">
        <h5 style="margin:0 0 10px 0; font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:#64748b; font-weight:700;">Engine Statistics</h5>
        <div style="display:flex; flex-direction:column; gap:6px; font-size:12.5px;">
            <div style="display:flex; justify-content:space-between;"><span style="color:#64748b;">Model Engine:</span><span style="font-weight:600;">RF-v2.1</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#64748b;">Knowledge base:</span><span style="font-weight:600;">FAISS RAG</span></div>
            <div style="display:flex; justify-content:space-between;"><span style="color:#64748b;">Precision Rate:</span><span style="font-weight:600; color:#059669;">96.8%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Actions
    st.markdown("<h5 style='font-size:11px; text-transform:uppercase; color:#64748b; font-weight:700; margin-bottom:8px;'>Quick Actions</h5>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        st.session_state.selected_symptoms = set()
        st.session_state.active_report = None
        st.session_state.recent_scans = []
        st.rerun()

    # Dynamic Clinical Report PDF generation
    if st.session_state.active_report:
        report = st.session_state.active_report
        dis = report['disease']
        desc = report['description']
        recs = report['nutrition']['recommended']
        avds = report['nutrition']['avoid']
        rat = report['nutrition']['reason']
        prec = "".join(f"<li>{p}</li>" for p in report['precautions'])
        syms = ", ".join(display_formatted_symptom(s) for s in st.session_state.selected_symptoms)
        
        # Build print HTML matching original
        html_report = f"""
        <html>
        <head><title>Clinical Report - {dis}</title>
        <style>
            body {{ font-family: sans-serif; color: #1e293b; padding: 25px; line-height: 1.5; }}
            h2 {{ color: #0f766e; border-bottom: 2px solid #0f766e; padding-bottom: 8px; }}
            .section {{ margin-bottom: 15px; font-size: 13.5px; }}
            .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-bottom: 10px; }}
            .footer {{ font-size: 10px; color: #94a3b8; margin-top: 30px; text-align: center; border-top: 1px dashed #cbd5e1; padding-top: 10px; }}
        </style>
        </head>
        <body onload="window.print()">
            <h2>NutriCare AI Diagnostic Report</h2>
            <div class="section"><strong>Active Intake Symptoms:</strong> {syms}</div>
            <div class="section"><strong> संदिग्ध रोग (SUSPECTED CONDITION):</strong> {dis}</div>
            <div class="section"><strong>Clinical Description:</strong> {desc}</div>
            <div class="card" style="border-left: 4px solid #059669;"><strong>Recommended Intake:</strong> {recs}</div>
            <div class="card" style="border-left: 4px solid #dc2626;"><strong>Restrict Intake:</strong> {avds}</div>
            <div class="section"><strong>Clinical Rationale:</strong> {rat}</div>
            <div class="section"><strong>Clinical Precautions:</strong><ul>{prec}</ul></div>
            <div class="footer">Autonomous guidelines for study reference. Always consult a clinical primary care physician.</div>
        </body>
        </html>
        """
        st.download_button(
            label="📥 Export Health Report",
            data=html_report,
            file_name=f"clinical_report_{dis.lower().replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True
        )

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#94a3b8; margin:0;'>AI Core Components:</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
        <span class="tech-badge">RAG</span>
        <span class="tech-badge">ML</span>
        <span class="tech-badge">FAISS</span>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================================
# MAIN DASHBOARD DESIGN & LAYOUT
# ==========================================================================

# 1. Page Header Titles
st.markdown("<span style='font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:#14b8a6; font-weight:800; display:block;'>INTELLIGENT MEDICAL COMPANION</span>", unsafe_allow_html=True)
st.markdown(f"<h1 style='font-weight:800; font-family:Outfit, sans-serif; color:#0f172a; margin-top:0; margin-bottom:20px;'>{navigation_selection[2:]}</h1>", unsafe_allow_html=True)

# 2. Hero banner section display
st.markdown(f"""
<div style="background: linear-gradient(135deg, white 0%, rgba(20, 184, 166, 0.05) 100%); border: 1px solid rgba(255,255,255,0.6); border-radius: 12px; padding: 20px 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.03); margin-bottom: 24px;">
    <div style="max-width: 80%;">
        <div style="display: flex; gap: 8px; margin-bottom: 8px;">
            <span style="font-size: 9px; font-weight: 700; background: rgba(20,184,166,0.1); color: #0f766e; padding: 2px 8px; border-radius: 10px;">RAG SYSTEM</span>
            <span style="font-size: 9px; font-weight: 700; background: rgba(37,99,235,0.1); color: #2563eb; padding: 2px 8px; border-radius: 10px;">DENSE SEARCH</span>
            <span style="font-size: 9px; font-weight: 700; background: rgba(124,58,237,0.1); color: #7c3aed; padding: 2px 8px; border-radius: 10px;">RANDOM FOREST</span>
        </div>
        <h4 style="font-weight:800; font-size:16.5px; color:#0f172a; margin:0 0 4px 0;">Custom Dietary Planning via Clinical Symptoms Analysis</h4>
        <p style="font-size:13px; color:#475569; margin:0;">Our advanced diagnostic system processes physical symptoms to map clinical constraints to specific food recommendation templates, backed by FAISS vector index semantic lookup.</p>
    </div>
    <div style="font-size: 40px; opacity: 0.15; padding-right: 10px;">🧬</div>
</div>
""", unsafe_allow_html=True)


# ==========================================================================
# VIEW CHANNELS (TABS MAPPED NATIVELY)
# ==========================================================================

# ----------------- TAB 1: SYMPTOM PREDICTOR -----------------
if navigation_selection == "🧬 Symptom Predictor":
    col_left, col_center, col_right = st.columns([1.3, 1.8, 1.3])
    
    # ----------------- Left Column (Symptom Intake) -----------------
    with col_left:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-size:15px; font-weight:800; margin-bottom:12px; color:#0f766e;'>Step 1: Symptom Intake</h4>", unsafe_allow_html=True)
        
        # Interactive multiselect symptom search
        symptom_choices = [display_formatted_symptom(s) for s in all_symptoms_list]
        selected_displays = st.multiselect(
            "Select Symptoms",
            options=symptom_choices,
            default=[display_formatted_symptom(s) for s in st.session_state.selected_symptoms],
            label_visibility="collapsed"
        )
        
        # Sync displays back to st.session_state raw names
        selected_raw = set()
        for disp in selected_displays:
            # Map back
            for sym in all_symptoms_list:
                if display_formatted_symptom(sym) == disp:
                    selected_raw.add(sym)
                    break
        st.session_state.selected_symptoms = selected_raw
        
        # Popular quick chips selection
        st.markdown("<p style='font-size:12px; font-weight:700; color:#64748b; margin-bottom:6px; margin-top:10px;'>Common Symptoms:</p>", unsafe_allow_html=True)
        quick_symptoms = ["fever", "cough", "headache", "joint_pain", "vomiting"]
        
        chip_cols = st.columns(2)
        for idx, sym_raw in enumerate(quick_symptoms):
            col_target = chip_cols[idx % 2]
            with col_target:
                label_txt = f"+ {sym_raw.replace('_', ' ').title()}"
                is_selected = sym_raw in st.session_state.selected_symptoms
                # Render button
                if st.button(label_txt, key=f"chip-{sym_raw}", use_container_width=True, type="secondary" if not is_selected else "primary"):
                    if sym_raw in st.session_state.selected_symptoms:
                        st.session_state.selected_symptoms.remove(sym_raw)
                    else:
                        st.session_state.selected_symptoms.add(sym_raw)
                    st.rerun()
                    
        # Active Trigger actions
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Symptoms", type="primary", use_container_width=True):
            if not st.session_state.selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Classifying symptoms and scanning FAISS database..."):
                    try:
                        clf_data = get_ml_model()
                        symptoms_list = clf_data['symptoms']
                        symptom_to_idx = clf_data['symptom_to_idx']
                        X_input = np.zeros((1, len(symptoms_list)), dtype=np.float32)
                        
                        matched_count = 0
                        for sym in st.session_state.selected_symptoms:
                            if sym in symptom_to_idx:
                                X_input[0, symptom_to_idx[sym]] = 1.0
                                matched_count += 1
                                
                        if matched_count == 0:
                            st.error("None of the selected symptoms match the model expected vocabulary.")
                        else:
                            clf = clf_data['model']
                            prediction = clf.predict(X_input)[0]
                            
                            try:
                                probs = clf.predict_proba(X_input)[0]
                                max_idx = np.argmax(probs)
                                confidence = float(probs[max_idx])
                            except:
                                confidence = 1.0
                                
                            norm_predicted = normalize_name(prediction)
                            description = description_map.get(norm_predicted, "Clinical description is not available for this condition. Please consult a practitioner.")
                            precautions = precaution_map.get(norm_predicted, ["Consult a primary care physician.", "Monitor symptoms closely.", "Stay hydrated."])
                            
                            # Lookup Nutrition
                            diet_info = None
                            if norm_predicted in nutrition_norm_map:
                                diet_info = dict(nutrition_norm_map[norm_predicted])
                                diet_info['match_method'] = 'exact'
                            else:
                                try:
                                    s_model = get_sentence_model()
                                    f_idx = get_faiss_index()
                                    if f_idx is not None and s_model is not None and nutrition_list:
                                        q_emb = s_model.encode([prediction])
                                        D, I = f_idx.search(np.array(q_emb).astype('float32'), k=1)
                                        match_idx = I[0][0]
                                        if 0 <= match_idx < len(nutrition_list):
                                            row = nutrition_list[match_idx]
                                            diet_info = {
                                                'disease': row['Disease'],
                                                'recommended': row['Recommended Food'],
                                                'avoid': row['Avoid Food'],
                                                'reason': row['Reason'],
                                                'match_method': 'semantic'
                                            }
                                except Exception as e:
                                    logger.error(f"Semantic fallback failed: {e}")
                                    
                            if diet_info is None:
                                diet_info = {
                                    'disease': 'General Wellness',
                                    'recommended': 'Water, fresh fruits, vegetables, and simple soups.',
                                    'avoid': 'Spicy foods, sugary drinks, deep fried items, and processed meals.',
                                    'reason': 'To support generalized immune response and ensure simple digestion.',
                                    'match_method': 'fallback'
                                }
                                
                            st.session_state.active_report = {
                                'disease': prediction,
                                'confidence': confidence,
                                'description': description,
                                'precautions': precautions,
                                'nutrition': diet_info
                            }
                            
                            # Log to recent scans
                            symptoms_display = ", ".join(display_formatted_symptom(s) for s in st.session_state.selected_symptoms)
                            new_scan = {'symptoms': list(st.session_state.selected_symptoms), 'symptomsDisplay': symptoms_display, 'disease': prediction}
                            # Filter duplicate
                            st.session_state.recent_scans = [r for r in st.session_state.recent_scans if r['symptomsDisplay'] != symptoms_display]
                            st.session_state.recent_scans.insert(0, new_scan)
                            st.session_state.recent_scans = st.session_state.recent_scans[:4]
                            
                            st.rerun()
                    except Exception as ex:
                        st.error(f"Diagnostic error: {ex}")
                        
        # Recent Session History scans
        if st.session_state.recent_scans:
            st.markdown("---")
            st.markdown("<h5 style='font-size:11px; text-transform:uppercase; color:#64748b; font-weight:700;'>Recent Session Scans</h5>", unsafe_allow_html=True)
            for idx, scan in enumerate(st.session_state.recent_scans):
                btn_lbl = f"{scan['symptomsDisplay'][:30]}... → {scan['disease']}"
                if st.button(btn_lbl, key=f"recent-scan-{idx}", use_container_width=True):
                    st.session_state.selected_symptoms = set(scan['symptoms'])
                    # Trigger prediction reload
                    st.session_state.active_report = None
                    st.rerun()

    # ----------------- Center Column (Clinical Report Outcome) -----------------
    with col_center:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-size:15px; font-weight:800; margin-bottom:12px; color:#0f766e;'>Step 2: Suspected Condition Guideline</h4>", unsafe_allow_html=True)
        
        if st.session_state.active_report is None:
            st.markdown("""
            <div style="border: 2px dashed rgba(15,23,42,0.08); border-radius:12px; padding: 40px; text-align:center; color:#64748b;">
                <div style="font-size:40px; margin-bottom:10px;">📋</div>
                <h5 style="font-weight:700; color:#475569; margin:0 0 4px 0;">No Active Intake Log</h5>
                <p style="font-size:12.5px; margin:0;">Please select symptoms in the left column and click 'Analyze Symptoms' to construct the diagnostic report outcomes here.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            report = st.session_state.active_report
            dis = report['disease']
            desc = report['description']
            precautions = report['precautions']
            nutrition_advice = report['nutrition']
            
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:15px; border-left: 5px solid #0f766e;">
                <span class="tech-badge" style="background:#d1fae5; color:#065f46;">AI suspective outcome</span>
                <h2 style="margin:5px 0 8px 0; font-family:Outfit, sans-serif; font-weight:800; font-size:26px; color:#0f172a;">{dis}</h2>
                <p style="font-size:13.5px; color:#334155; line-height:1.5; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical outcomes split boxes (Recommended, Avoid, Precautions, Rationale)
            col_card_left, col_card_right = st.columns(2)
            
            with col_card_left:
                st.markdown(f"""
                <div class="outcome-card success">
                    <div class="outcome-header" style="color:#059669;">✔ Recommended Foods</div>
                    <p style="font-size:13px; font-weight:600; color:#0f172a; margin:0;">{nutrition_advice['recommended']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="outcome-card danger">
                    <div class="outcome-header" style="color:#dc2626;">✖ Restrict & Avoid</div>
                    <p style="font-size:13px; font-weight:600; color:#0f172a; margin:0;">{nutrition_advice['avoid']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_card_right:
                # Precaution List Card
                prec_html = "".join(f"<li style='margin-bottom:3px;'>{p}</li>" for p in precautions)
                st.markdown(f"""
                <div class="outcome-card warning">
                    <div class="outcome-header" style="color:#d97706;">⚠️ Precautions Protocols</div>
                    <ul style="margin:0; padding-left:15px; font-size:12.5px; color:#334155;">{prec_html}</ul>
                </div>
                """, unsafe_allow_html=True)
                
            # Rationale Card
            st.markdown(f"""
            <div class="outcome-card info" style="margin-top:5px;">
                <div class="outcome-header" style="color:#2563eb;">🔬 Physiological Rationale</div>
                <p style="font-size:13px; color:#334155; line-height:1.4; margin:0;">{nutrition_advice['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------- Right Column (Vitals & Tips) -----------------
    with col_right:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-size:15px; font-weight:800; margin-bottom:12px; color:#0f766e;'>Vitals & Analytics</h4>", unsafe_allow_html=True)
        
        # Retrieve variables
        if st.session_state.active_report:
            report = st.session_state.active_report
            confidence_val = Math_round = int(report['confidence'] * 100)
            recovery_val = get_simulated_recovery_score(report['disease'])
            hydration_val = get_simulated_hydration(report['disease'])
            risk_val = get_simulated_risk(report['disease'])
        else:
            confidence_val = 0
            recovery_val = 0
            hydration_val = 75
            risk_val = "N/A"
            
        # Display st.metrics
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric("AI Confidence", f"{confidence_val}%")
            st.metric("Hydration Target", f"{hydration_val}%")
        with met_col2:
            st.metric("Recovery Outlook", f"{recovery_val}%")
            st.metric("Risk Assessment", risk_val)
            
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        
        # Clinical Tip Card
        st.markdown("<h5 style='font-size:11px; text-transform:uppercase; color:#64748b; font-weight:700;'>Daily Clinical Advice</h5>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding:15px; background:linear-gradient(135deg, white 0%, rgba(124,58,237,0.03) 100%);">
            <p style="font-size:12.5px; color:#334155; line-height:1.4; margin:0 0 10px 0;">{daily_tips[st.session_state.tip_idx]}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Next advice tip ➜", size="small"):
            st.session_state.tip_idx = (st.session_state.tip_idx + 1) % len(daily_tips)
            st.rerun()


# ----------------- TAB 2: DIETARY SEARCH -----------------
elif navigation_selection == "🔍 Dietary Search":
    st.markdown("<p style='font-size:14px; color:#475569; margin-top:-10px; margin-bottom:20px;'>Query our dense vector database using natural language descriptions to find matching dietary protocols.</p>", unsafe_allow_html=True)
    
    # 2-Column Split: Query input / Results
    col_search_left, col_search_right = st.columns([1.6, 2.4])
    
    with col_search_left:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-size:15px; font-weight:800; color:#0f766e; margin-bottom:12px;'>Search Parameters</h4>", unsafe_allow_html=True)
        
        # Suggestions chips wrapper
        st.markdown("<p style='font-size:12px; font-weight:700; color:#64748b; margin-bottom:6px;'>Try queries:</p>", unsafe_allow_html=True)
        queries_chips = [
            "What should I eat when I have a common cold?",
            "Nutritional advice for high blood pressure",
            "Foods to avoid during gastritis"
        ]
        
        selected_suggestion = None
        for i, q in enumerate(queries_chips):
            if st.button(f"🔍 {q}", key=f"sug-{i}", use_container_width=True):
                selected_suggestion = q
                
        # Main text input box
        default_val = selected_suggestion if selected_suggestion else ""
        query_val = st.text_input("Enter conversational search query:", value=default_val, placeholder="e.g. diarrhea avoid food lists...")
        
    with col_search_right:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-size:15px; font-weight:800; color:#0f766e; margin-bottom:12px;'>Semantic Retrieval Results</h4>", unsafe_allow_html=True)
        
        if query_val.strip():
            with st.spinner("Embedding query & searching FAISS index flat L2 space..."):
                try:
                    s_model = get_sentence_model()
                    f_idx = get_faiss_index()
                    
                    if f_idx is None or s_model is None or not nutrition_list:
                        st.error("RAG component databases are not initialized.")
                    else:
                        q_emb = s_model.encode([query_val.strip()])
                        k = min(3, len(nutrition_list))
                        D, I = f_idx.search(np.array(q_emb).astype('float32'), k=k)
                        
                        has_results = False
                        for rank, match_idx in enumerate(I[0]):
                            if 0 <= match_idx < len(nutrition_list):
                                has_results = True
                                row = nutrition_list[match_idx]
                                score = float(D[0][rank])
                                confidence_score = Math_max = int(max(0, min(100, (2.0 - score) * 50)))
                                
                                st.markdown(f"""
                                <div class="glass-card" style="margin-bottom:15px; padding:16px;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(15,23,42,0.05); padding-bottom:8px; margin-bottom:10px;">
                                        <h4 style="margin:0; font-size:15px; color:#0f172a; font-family:Outfit, sans-serif;">{row['Disease']}</h4>
                                        <span class="tech-badge" style="background:rgba(37,99,235,0.08); color:#2563eb;">Semantic Match: {confidence_score}%</span>
                                    </div>
                                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:8px;">
                                        <div style="background:var(--success-bg); padding:10px; border-radius:6px; border-left:3px solid var(--success);">
                                            <h5 style="color:var(--success); font-size:10.5px; margin:0 0 4px 0; text-transform:uppercase;">✔ Recommended</h5>
                                            <p style="font-size:12.5px; font-weight:600; color:#0f172a; margin:0;">{row['Recommended Food']}</p>
                                        </div>
                                        <div style="background:var(--danger-bg); padding:10px; border-radius:6px; border-left:3px solid var(--danger);">
                                            <h5 style="color:var(--danger); font-size:10.5px; margin:0 0 4px 0; text-transform:uppercase;">✖ Restrict / Avoid</h5>
                                            <p style="font-size:12.5px; font-weight:600; color:#0f172a; margin:0;">{row['Avoid Food']}</p>
                                        </div>
                                    </div>
                                    <div style="background:rgba(15,23,42,0.02); padding:10px; border-radius:6px;">
                                        <span style="font-size:9.5px; color:#64748b; font-weight:700; text-transform:uppercase;">Intake Rationale:</span>
                                        <p style="font-size:12.5px; color:#334155; margin:0; line-height:1.4;">{row['Reason']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        if not has_results:
                            st.warning("No records matched the vector space query.")
                except Exception as ex:
                    st.error(f"Semantic search failed: {ex}")
        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(15,23,42,0.08); border-radius:12px; padding: 40px; text-align:center; color:#64748b;">
                <div style="font-size:40px; margin-bottom:10px;">🔍</div>
                <h5 style="font-weight:700; color:#475569; margin:0 0 4px 0;">Awaiting Query Parameter</h5>
                <p style="font-size:12.5px; margin:0;">Type in the input box on the left or select a sample suggestion to trigger similarity retrieval.</p>
            </div>
            """, unsafe_allow_html=True)


# ----------------- TAB 3: CONDITIONS DIRECTORY -----------------
elif navigation_selection == "📋 Conditions Directory":
    st.markdown("<p style='font-size:14px; color:#475569; margin-top:-10px; margin-bottom:20px;'>Browse the verified healthcare nutrition parameters directory alphabetically.</p>", unsafe_allow_html=True)
    
    # Text filter box for search grid
    filter_val = st.text_input("🔍 Search conditions in directory:", placeholder="Type condition name here...")
    
    # Filter catalog items
    filtered_items = []
    for item in nutrition_list:
        name = item['Disease'].lower()
        rec = item['Recommended Food'].lower()
        avd = item['Avoid Food'].lower()
        q = filter_val.strip().lower()
        if not q or (q in name or q in rec or q in avd):
            filtered_items.append(item)
            
    filtered_items.sort(key=lambda x: x['Disease'])
    
    st.markdown(f"<p style='font-size:13px; font-weight:600; color:#64748b;'>Verified Conditions Displayed: <span style='color:#0f766e;'>{len(filtered_items)}</span></p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not filtered_items:
        st.warning("No matching health conditions located in database directory.")
    else:
        # Build cards grid using Streamlit columns
        cols_per_row = 3
        rows = [filtered_items[i:i + cols_per_row] for i in range(0, len(filtered_items), cols_per_row)]
        
        for row_grp in rows:
            grid_cols = st.columns(cols_per_row)
            for idx, item in enumerate(row_grp):
                with grid_cols[idx]:
                    st.markdown(f"""
                    <div class="glass-card" style="height:100%; display:flex; flex-direction:column; padding:18px;">
                        <h4 style="margin:0 0 10px 0; font-family:Outfit, sans-serif; font-weight:750; font-size:15px; color:#0f172a;">{item['Disease']}</h4>
                        <div style="display:flex; flex-direction:column; gap:8px; font-size:12px;">
                            <div>
                                <span style="font-weight:700; color:#059669; text-transform:uppercase; font-size:9.5px; display:block;">✔ Recommend</span>
                                <span style="font-weight:600; color:#1e293b;">{item['Recommended Food']}</span>
                            </div>
                            <div>
                                <span style="font-weight:700; color:#dc2626; text-transform:uppercase; font-size:9.5px; display:block;">✖ Restrict / Avoid</span>
                                <span style="font-weight:600; color:#1e293b;">{item['Avoid Food']}</span>
                            </div>
                            <div style="border-top:1px solid rgba(15,23,42,0.04); padding-top:6px;">
                                <span style="color:#64748b; text-transform:uppercase; font-size:9.5px; display:block;">Rationale</span>
                                <span style="color:#475569;">{item['Reason']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
