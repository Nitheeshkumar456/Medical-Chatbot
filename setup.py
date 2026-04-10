"""
Run this once before starting the Streamlit app:
    python setup.py
"""
import os
import sys

print("=" * 55)
print("  MedBot — Setup Script")
print("=" * 55)

print("\n[1/2] Generating dataset...")
exec(open("data/generate_dataset.py").read())

print("\n[2/2] Training ML models...")
from train_model import train_models
train_models()

print("\n✅ Setup complete!")
print("\nTo launch the app, run:")
print("    streamlit run app.py")
