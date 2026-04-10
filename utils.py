import re
import pickle

# Disease info database
DISEASE_INFO = {
    "Flu": {
        "description": "Influenza (flu) is a contagious respiratory illness caused by influenza viruses.",
        "precautions": ["Rest and sleep", "Stay hydrated", "Take fever-reducing medication", "Avoid contact with others"],
        "severity": "Moderate",
        "specialist": "General Physician",
    },
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract, usually harmless.",
        "precautions": ["Drink plenty of fluids", "Get rest", "Use saline nasal drops", "Avoid cold air"],
        "severity": "Mild",
        "specialist": "General Physician",
    },
    "COVID-19": {
        "description": "An infectious disease caused by the SARS-CoV-2 virus affecting the respiratory system.",
        "precautions": ["Isolate immediately", "Wear a mask", "Contact healthcare provider", "Monitor oxygen levels"],
        "severity": "High",
        "specialist": "Infectious Disease Specialist",
    },
    "Diabetes": {
        "description": "A metabolic disease that causes high blood sugar due to insulin issues.",
        "precautions": ["Monitor blood sugar", "Follow diabetic diet", "Exercise regularly", "Take prescribed medication"],
        "severity": "High",
        "specialist": "Endocrinologist",
    },
    "Hypertension": {
        "description": "A condition where the force of blood against artery walls is consistently too high.",
        "precautions": ["Reduce salt intake", "Exercise regularly", "Limit alcohol", "Take blood pressure medication"],
        "severity": "High",
        "specialist": "Cardiologist",
    },
    "Malaria": {
        "description": "A life-threatening disease caused by Plasmodium parasites transmitted by mosquitoes.",
        "precautions": ["Take antimalarial drugs", "Use mosquito nets", "Apply insect repellent", "Seek immediate medical care"],
        "severity": "High",
        "specialist": "Infectious Disease Specialist",
    },
    "Typhoid": {
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food/water.",
        "precautions": ["Take prescribed antibiotics", "Drink clean water only", "Avoid raw foods", "Rest adequately"],
        "severity": "High",
        "specialist": "General Physician",
    },
    "Dengue": {
        "description": "A mosquito-borne viral infection causing flu-like illness and potentially severe complications.",
        "precautions": ["Stay hydrated", "Use mosquito repellent", "Monitor platelet count", "Seek immediate care if severe"],
        "severity": "High",
        "specialist": "Infectious Disease Specialist",
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs in one or both lungs.",
        "precautions": ["Take prescribed antibiotics", "Rest completely", "Stay hydrated", "Seek emergency care if breathing worsens"],
        "severity": "High",
        "specialist": "Pulmonologist",
    },
    "Asthma": {
        "description": "A chronic condition in which airways narrow and swell, producing extra mucus.",
        "precautions": ["Use inhaler as prescribed", "Avoid triggers", "Monitor peak flow", "Keep rescue inhaler handy"],
        "severity": "Moderate",
        "specialist": "Pulmonologist",
    },
    "Migraine": {
        "description": "A neurological condition causing intense, debilitating headaches often with nausea.",
        "precautions": ["Rest in dark quiet room", "Apply cold compress", "Take pain relievers", "Avoid known triggers"],
        "severity": "Moderate",
        "specialist": "Neurologist",
    },
    "Anemia": {
        "description": "A condition where you lack enough healthy red blood cells to carry adequate oxygen.",
        "precautions": ["Take iron supplements", "Eat iron-rich foods", "Avoid tea/coffee with meals", "Regular blood tests"],
        "severity": "Moderate",
        "specialist": "Hematologist",
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, typically caused by viral or bacterial infection.",
        "precautions": ["Stay hydrated with ORS", "Rest", "Eat bland foods (BRAT diet)", "Wash hands frequently"],
        "severity": "Moderate",
        "specialist": "Gastroenterologist",
    },
    "Appendicitis": {
        "description": "Inflammation of the appendix, a medical emergency requiring prompt treatment.",
        "precautions": ["Seek emergency medical care immediately", "Do not eat or drink", "Avoid pain medication before diagnosis", "Surgery likely needed"],
        "severity": "Critical",
        "specialist": "General Surgeon",
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system — kidneys, bladder, or urethra.",
        "precautions": ["Drink plenty of water", "Take prescribed antibiotics", "Urinate frequently", "Avoid irritants like caffeine"],
        "severity": "Moderate",
        "specialist": "Urologist",
    },
}

SEVERITY_COLORS = {
    "Mild": "🟢",
    "Moderate": "🟡",
    "High": "🔴",
    "Critical": "🆘",
}

def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/symptoms_list.pkl", "rb") as f:
        symptoms = pickle.load(f)
    return model, le, symptoms

def extract_symptoms_from_text(text, symptoms_list):
    text = text.lower()
    text = re.sub(r"[^a-z\s_]", " ", text)
    
    # synonym map
    synonyms = {
        "temperature": "fever", "high temperature": "fever", "hot": "fever",
        "tired": "fatigue", "exhausted": "fatigue", "weak": "weakness",
        "ache": "body_ache", "aching": "body_ache", "pain": "body_ache",
        "runny nose": "runny_nose", "stuffy": "congestion", "stuffed": "congestion",
        "sneeze": "sneezing", "sneezes": "sneezing",
        "breath": "shortness_of_breath", "breathing": "shortness_of_breath",
        "taste": "loss_of_taste", "smell": "loss_of_smell",
        "thirst": "excessive_thirst", "thirsty": "excessive_thirst",
        "urinate": "frequent_urination", "urination": "frequent_urination",
        "dizzy": "dizziness", "lightheaded": "dizziness",
        "nauseous": "nausea", "sick feeling": "nausea",
        "throw up": "vomiting", "throwing up": "vomiting",
        "stomach": "abdominal_pain", "belly": "abdominal_pain",
        "loose stool": "diarrhea", "loose motion": "diarrhea",
        "eye pain": "eye_pain", "joint": "joint_pain",
        "rashes": "rash", "spots": "rash",
        "chest tight": "chest_tightness", "tightness": "chest_tightness",
        "wheeze": "wheezing", "whistling": "wheezing",
        "pale": "pale_skin", "paleness": "pale_skin",
        "cold hands": "cold_hands", "cold feet": "cold_hands",
    }

    detected = []
    for synonym, symptom in synonyms.items():
        if synonym in text and symptom in symptoms_list:
            detected.append(symptom)

    for symptom in symptoms_list:
        readable = symptom.replace("_", " ")
        if readable in text or symptom in text:
            detected.append(symptom)

    return list(set(detected))

def predict_disease(selected_symptoms, model, le, symptoms_list):
    input_vector = [1 if s in selected_symptoms else 0 for s in symptoms_list]
    import numpy as np
    input_array = np.array(input_vector).reshape(1, -1)
    
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    
    disease_name = le.inverse_transform([prediction])[0]
    confidence = max(probabilities) * 100

    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [(le.inverse_transform([i])[0], round(probabilities[i] * 100, 1)) for i in top_indices]

    return disease_name, confidence, top_predictions
