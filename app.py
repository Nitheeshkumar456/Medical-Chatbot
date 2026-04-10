import streamlit as st
import numpy as np
import pandas as pd
from utils import (
    load_artifacts, extract_symptoms_from_text,
    predict_disease, DISEASE_INFO, SEVERITY_COLORS
)

st.set_page_config(
    page_title="MedBot — AI Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-user {
        background: #e3f2fd;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .chat-bot {
        background: #f1f8e9;
        border-left: 4px solid #43a047;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .disease-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .symptom-tag {
        background: #e3f2fd;
        color: #1a73e8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .disclaimer {
        background: #fff3e0;
        border: 1px solid #ffb300;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #e65100;
    }
    [data-testid="stSidebar"] { background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load_artifacts()

model, le, symptoms_list = load_model()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=60)
    st.title("🏥 MedBot")
    st.markdown("*AI-Powered Disease Predictor*")
    st.divider()

    mode = st.radio(
        "Input Mode",
        ["💬 Chat with MedBot", "✅ Symptom Checklist"],
        index=0,
    )
    st.session_state.mode = "chat" if "Chat" in mode else "checklist"

    st.divider()
    st.markdown("### About")
    st.info(
        "MedBot uses Machine Learning (Random Forest, Decision Tree, Naive Bayes) "
        "to predict possible diseases from your symptoms.\n\n"
        "**Not a substitute for professional medical advice.**"
    )

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_symptoms = []
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Chatbot</h1>
    <p>Describe your symptoms and get an AI-powered disease prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    Always consult a qualified healthcare professional for medical diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "chat":
    st.subheader("💬 Chat with MedBot")

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 Hello! I'm **MedBot**, your AI medical assistant.\n\n"
                "Please describe your symptoms in plain English — for example:\n"
                "> *'I have fever, headache, and body ache since yesterday'*\n\n"
                "I'll analyze your symptoms and suggest possible conditions."
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Describe your symptoms here…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        detected = extract_symptoms_from_text(user_input, symptoms_list)

        if len(detected) < 2:
            response = (
                "🤔 I couldn't detect enough symptoms from your message.\n\n"
                "Try describing symptoms like: *fever, headache, cough, fatigue, nausea, body ache*...\n\n"
                "Or switch to **Symptom Checklist** mode in the sidebar for easier selection."
            )
        else:
            disease, confidence, top3 = predict_disease(detected, model, le, symptoms_list)
            info = DISEASE_INFO.get(disease, {})
            severity = info.get("severity", "Unknown")
            severity_icon = SEVERITY_COLORS.get(severity, "⚪")
            precautions = info.get("precautions", [])
            specialist = info.get("specialist", "General Physician")
            description = info.get("description", "")

            sym_tags = "".join(f'<span class="symptom-tag">✓ {s.replace("_"," ")}</span>' for s in detected)

            precaution_list = "".join(f"<li>{p}</li>" for p in precautions)

            top3_rows = "".join(
                f"<tr><td style='padding:4px 8px'>{d}</td>"
                f"<td><div style='background:#e3f2fd;width:{min(c,100):.0f}%;height:14px;border-radius:4px'></div></td>"
                f"<td style='padding:4px 8px'><b>{c:.1f}%</b></td></tr>"
                for d, c in top3
            )

            response = f"""
<div class="disease-card">
<h3>🩺 Diagnosis Result</h3>

<b>Symptoms Detected:</b><br>{sym_tags}<br><br>

<table width="100%"><tr>
<td width="50%">
<div class="metric-card">
<div style="font-size:2rem">{severity_icon}</div>
<div style="font-size:1.4rem;font-weight:700;color:#1a73e8">{disease}</div>
<div style="color:#666">Predicted Disease</div>
</div>
</td>
<td width="50%" style="padding-left:12px">
<div class="metric-card">
<div style="font-size:2rem">🎯</div>
<div style="font-size:1.4rem;font-weight:700;color:#43a047">{confidence:.1f}%</div>
<div style="color:#666">Confidence Score</div>
</div>
</td>
</tr></table>

<br><b>📋 About:</b> {description}

<br><br><b>📊 Top Predictions:</b>
<table width="100%" style="margin-top:8px">{top3_rows}</table>

<br><b>🛡️ Recommended Precautions:</b>
<ol>{precaution_list}</ol>

<b>👨‍⚕️ Consult:</b> {specialist} &nbsp; | &nbsp; <b>Severity:</b> {severity_icon} {severity}
</div>
"""

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("✅ Symptom Checklist")

    # group symptoms visually
    categories = {
        "🌡️ Fever & Temperature": ["fever", "high_fever", "mild_fever", "chills", "sweating"],
        "😮‍💨 Respiratory": ["cough", "shortness_of_breath", "wheezing", "chest_tightness", "congestion"],
        "🤕 Head & Neuro": ["headache", "severe_headache", "dizziness", "light_sensitivity", "sound_sensitivity", "blurred_vision"],
        "🤢 Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_of_appetite", "bloating", "dehydration"],
        "💪 Body & Muscles": ["body_ache", "joint_pain", "fatigue", "weakness", "chest_pain"],
        "🩺 Other Symptoms": [s for s in symptoms_list if s not in [
            "fever","high_fever","mild_fever","chills","sweating",
            "cough","shortness_of_breath","wheezing","chest_tightness","congestion",
            "headache","severe_headache","dizziness","light_sensitivity","sound_sensitivity","blurred_vision",
            "nausea","vomiting","diarrhea","abdominal_pain","loss_of_appetite","bloating","dehydration",
            "body_ache","joint_pain","fatigue","weakness","chest_pain"
        ]],
    }

    selected = []
    cols = st.columns(2)
    cat_items = list(categories.items())
    for idx, (cat_name, cat_symptoms) in enumerate(cat_items):
        with cols[idx % 2]:
            st.markdown(f"**{cat_name}**")
            for symptom in cat_symptoms:
                label = symptom.replace("_", " ").title()
                if st.checkbox(label, key=f"chk_{symptom}"):
                    selected.append(symptom)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Selected symptoms:** {len(selected)}")
        if selected:
            tags = " ".join(f"`{s.replace('_',' ')}`" for s in selected)
            st.markdown(tags)

    with col2:
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True, type="primary")

    if predict_btn:
        if len(selected) < 2:
            st.warning("⚠️ Please select at least 2 symptoms for a prediction.")
        else:
            disease, confidence, top3 = predict_disease(selected, model, le, symptoms_list)
            info = DISEASE_INFO.get(disease, {})
            severity = info.get("severity", "Unknown")
            severity_icon = SEVERITY_COLORS.get(severity, "⚪")

            st.divider()
            st.subheader("🩺 Prediction Results")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Disease", disease)
            m2.metric("Confidence", f"{confidence:.1f}%")
            m3.metric("Severity", f"{severity_icon} {severity}")

            tab1, tab2, tab3 = st.tabs(["📋 Details", "📊 Top Predictions", "🛡️ Precautions"])

            with tab1:
                st.markdown(f"**About {disease}:**")
                st.info(info.get("description", ""))
                st.markdown(f"**Recommended Specialist:** 👨‍⚕️ {info.get('specialist','General Physician')}")

            with tab2:
                df_top = pd.DataFrame(top3, columns=["Disease", "Confidence (%)"])
                st.bar_chart(df_top.set_index("Disease"))
                st.dataframe(df_top, use_container_width=True)

            with tab3:
                for i, p in enumerate(info.get("precautions", []), 1):
                    st.markdown(f"{i}. {p}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#999;font-size:0.8rem'>"
    "MedBot v1.0 · Built with Streamlit & scikit-learn · "
    "⚠️ For educational use only — not a substitute for professional medical advice"
    "</center>",
    unsafe_allow_html=True,
)
