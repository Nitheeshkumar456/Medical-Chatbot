# 🏥 MedBot — AI Medical Chatbot

An AI-powered medical chatbot that predicts diseases from user-described symptoms using Machine Learning classification models, deployed with Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)

---

## 🚀 Features

- **Two input modes** — free-text chat OR symptom checklist
- **NLP symptom extraction** — detects symptoms from natural language using synonym mapping + pattern matching
- **3 ML models** — Decision Tree, Naive Bayes, Random Forest (best model auto-selected)
- **98%+ accuracy** — Random Forest achieves ~98.9% on test data
- **Detailed results** — disease description, severity level, top 3 predictions, precautions, recommended specialist
- **15 diseases** covered — Flu, COVID-19, Diabetes, Malaria, Dengue, Pneumonia, and more

---

## 🗂️ Project Structure

```
medical-chatbot/
├── app.py                  # Streamlit application (main UI)
├── utils.py                # NLP extractor, predictor, disease info database
├── train_model.py          # Model training script
├── setup.py                # One-time setup (generate data + train models)
├── requirements.txt
├── data/
│   ├── generate_dataset.py # Synthetic dataset generator
│   └── medical_data.csv    # Generated dataset (1,800 rows, 44 symptoms)
├── models/
│   ├── best_model.pkl      # Trained Random Forest model
│   ├── label_encoder.pkl
│   ├── symptoms_list.pkl
│   └── all_models.pkl
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-chatbot.git
cd medical-chatbot
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run setup (generates data + trains models)
```bash
python setup.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🌐 Deploy on Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Add a `packages.txt` if needed (usually not required)
6. Click **Deploy** — your app goes live in ~2 minutes!

> **Note:** Streamlit Cloud runs `setup.py` implicitly if models don't exist.  
> Add this to the top of `app.py` as a fallback:
> ```python
> if not os.path.exists("models/best_model.pkl"):
>     exec(open("setup.py").read())
> ```

---

## 🧠 Machine Learning Details

| Model | Test Accuracy | CV Mean |
|-------|-------------|---------|
| Decision Tree | 80.00% | 78.61% ± 2.09% |
| Naive Bayes | 82.78% | 83.67% ± 2.44% |
| **Random Forest** | **98.89%** | **99.06% ± 0.42%** |

### Techniques Used
- **Feature Engineering** — Binary symptom encoding (presence/absence)
- **NLP** — Synonym mapping + regex pattern matching for free-text input
- **Cross-validation** — 5-fold CV for robust model evaluation
- **Label Encoding** — For multi-class disease labels
- **Auto model selection** — Best model saved automatically

---

## 🦠 Diseases Covered

| Disease | Severity | Specialist |
|---------|----------|------------|
| Flu | Moderate | General Physician |
| Common Cold | Mild | General Physician |
| COVID-19 | High | Infectious Disease Specialist |
| Diabetes | High | Endocrinologist |
| Hypertension | High | Cardiologist |
| Malaria | High | Infectious Disease Specialist |
| Typhoid | High | General Physician |
| Dengue | High | Infectious Disease Specialist |
| Pneumonia | High | Pulmonologist |
| Asthma | Moderate | Pulmonologist |
| Migraine | Moderate | Neurologist |
| Anemia | Moderate | Hematologist |
| Gastroenteritis | Moderate | Gastroenterologist |
| Appendicitis | Critical | General Surgeon |
| Urinary Tract Infection | Moderate | Urologist |

---

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

MIT License — free to use, modify, and distribute.
