import pandas as pd
import numpy as np

diseases = {
    "Flu": ["fever", "cough", "fatigue", "body_ache", "headache", "chills", "sore_throat"],
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough", "mild_fever", "congestion"],
    "COVID-19": ["fever", "cough", "fatigue", "loss_of_taste", "loss_of_smell", "shortness_of_breath", "body_ache"],
    "Diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing", "weight_loss"],
    "Hypertension": ["headache", "dizziness", "chest_pain", "shortness_of_breath", "blurred_vision", "nausea"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "vomiting", "body_ache"],
    "Typhoid": ["fever", "abdominal_pain", "headache", "fatigue", "diarrhea", "loss_of_appetite", "rash"],
    "Dengue": ["high_fever", "severe_headache", "eye_pain", "joint_pain", "rash", "bleeding", "nausea"],
    "Pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue", "sweating", "chills"],
    "Asthma": ["shortness_of_breath", "wheezing", "cough", "chest_tightness", "fatigue"],
    "Migraine": ["severe_headache", "nausea", "vomiting", "light_sensitivity", "sound_sensitivity", "blurred_vision"],
    "Anemia": ["fatigue", "weakness", "pale_skin", "dizziness", "shortness_of_breath", "cold_hands"],
    "Gastroenteritis": ["diarrhea", "vomiting", "nausea", "abdominal_pain", "fever", "dehydration"],
    "Appendicitis": ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite", "bloating"],
    "Urinary Tract Infection": ["frequent_urination", "burning_urination", "abdominal_pain", "fever", "cloudy_urine"],
}

all_symptoms = sorted(set(s for symptoms in diseases.values() for s in symptoms))

rows = []
np.random.seed(42)
for disease, core_symptoms in diseases.items():
    for _ in range(120):
        row = {s: 0 for s in all_symptoms}
        n = np.random.randint(max(2, len(core_symptoms) - 2), len(core_symptoms) + 1)
        chosen = np.random.choice(core_symptoms, size=min(n, len(core_symptoms)), replace=False)
        for s in chosen:
            row[s] = 1
        # add 0-2 random noise symptoms
        noise_pool = [s for s in all_symptoms if s not in core_symptoms]
        noise_count = np.random.randint(0, 3)
        for s in np.random.choice(noise_pool, size=min(noise_count, len(noise_pool)), replace=False):
            row[s] = 1
        row["disease"] = disease
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/medical_data.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Symptoms: {len(all_symptoms)}")
print(f"Diseases: {df['disease'].nunique()}")
