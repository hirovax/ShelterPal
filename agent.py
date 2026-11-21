import os

#import google.generativeai as genai
#from google.adk.agents import LlmAgent

from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

from clean_data import feature_engineering_days

from dotenv import load_dotenv

MODELS_DIR = Path(__file__).resolve().parent / "models"

def load_model(animal_type):
    animal_type = animal_type.lower()

    paths = {
        "dog": MODELS_DIR / "catboost_dogs",
        "cat": MODELS_DIR / "catboost_cats"
    }

    if animal_type not in paths:
        raise ValueError("Animal type must be 'dog' or 'cat' ")

    model = CatBoostClassifier()
    model.load_model(str[paths[animal_type]])

    return model

def predict_animal(animal_type, animal_data):
    model = load_model(animal_type)

    df = pd.DataFrame([animal_data])
    df_fe = feature_engineering_days(df).copy()

    if df_fe.empty:
        raise ValueError("After feature engineering the df is empty")

    X = df[model.feature_names_].copy()
    probs = model.predict_proba(X)[0]

    probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(model.classes_, probs)
    }

    return {
        "under_7_days": probabilities[0],
        "between_7_30_days": probabilities[1],
        "over_30_days": probabilities[2]
    }
    
def explain_animal():
    print("shap")

def get_animal_info():
    print("animal-info")

def main():
    load_dotenv()
    api_key = os.environ("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env")

    #client = genai.configure(api_key=api_key)
    
