import os

from google import genai
from google.genai import types

from pathlib import Path
import pandas as pd
import shap 

from catboost import CatBoostClassifier

from clean_data import feature_engineering_days

from dotenv import load_dotenv

MODELS_DIR = Path(__file__).resolve().parent / "models"

_CACHE = {}

def load_model(animal_type):
    animal_type = animal_type.lower()

    paths = {
        "dog": MODELS_DIR / "dog_days.cbm",
        "cat": MODELS_DIR / "cat_days.cbm"
    }

    if animal_type not in paths:
        raise ValueError("Animal type must be 'dog' or 'cat' ")

    if animal_type in _CACHE:
        return _CACHE[animal_type]
    
    model = CatBoostClassifier()
    model.load_model(str(paths[animal_type]))
    _CACHE[animal_type] = model

    return model



RAW_DIR = Path(__file__).resolve().parent / "data" / "austin" / "raw"

def load_historical_data(animal_type: str) -> pd.DataFrame:
    """Loads and caches data from acc_intakes_outcomes.csv"""
    animal_key = animal_type.strip().capitalize()  # 'Dog' lub 'Cat'
    cache_key = f"history_{animal_key.lower()}"

    if cache_key in _CACHE:
        return _CACHE[cache_key]

    file_path = RAW_DIR / "aac_intakes_outcomes.csv"

    df = pd.read_csv(
        file_path,
        usecols=["animal_type", "breed", "intake_type", "time_in_shelter_days"]
    )

    df = df[df["animal_type"] == animal_key].dropna(subset=["time_in_shelter_days"]).copy()

    df["breed_clean"] = df["breed"].astype(str).str.lower().str.strip()
    df["intake_type_clean"] = df["intake_type"].astype(str).str.lower().str.strip()

    _CACHE[cache_key] = df
    return df


def query_history(
        animal_type: str,
        breed: str = "",
        intake_type: str = "",
        min_sample_size: int = 10
    ) -> dict:
    """Aggregates historical shelter stay statistics, medians, percentiles, 
    and long-stay rates for a given animal segment, breed, or intake type.

    Args:
        animal_type: 'dog' or 'cat'
        breed: Breed name or substring (e.g., 'Pit Bull', 'Labrador Retriever', 'Domestic Shorthair')
        intake_type: Intake reason (e.g., 'Stray', 'Owner Surrender', 'Public Assist')
        min_sample_size: Minimum records required for specific breed/intake slice (default 10)
    """
    print(">>> [TOOL EXECUTION] query_history has been called!")

    clean_animal = animal_type.strip().lower()
    if clean_animal not in ["dog", "cat"]:
        return {"error": "animal_type must be 'dog' or 'cat'"}

    df = load_historical_data(clean_animal)
    filtered = df
    applied_filters = {"animal_type": clean_animal}

    # filtering by breed
    if breed:
        breed_term = breed.strip().lower()
        breed_subset = filtered[filtered["breed_clean"].str.contains(breed_term, na=False, regex=False)]
        
        if len(breed_subset) >= min_sample_size:
            filtered = breed_subset
            applied_filters["breed"] = breed
        else:
            applied_filters["breed_note"] = (
                f"Only {len(breed_subset)} cases found for '{breed}'. "
                f"Showing broader statistics for all {clean_animal}s instead."
            )

    # filtering by intake type
    if intake_type:
        intake_term = intake_type.strip().lower()
        intake_subset = filtered[filtered["intake_type_clean"] == intake_term]
        
        if len(intake_subset) >= min_sample_size:
            filtered = intake_subset
            applied_filters["intake_type"] = intake_type
        else:
            applied_filters["intake_type_note"] = (
                f"Insufficient sample ({len(intake_subset)}) for intake_type '{intake_type}'."
            )

    sample_size = len(filtered)
    if sample_size == 0:
        return {"error": "No historical cases matched these criteria.", "applied_filters": applied_filters}

    stay = filtered["time_in_shelter_days"]

    # calculating key metrics for catboost
    pct_under_4 = float((stay <= 4.0).mean() * 100)
    pct_4_to_10 = float(((stay > 4.0) & (stay <= 10.0)).mean() * 100)
    pct_over_10 = float((stay > 10.0).mean() * 100)

    return {
        "applied_filters": applied_filters,
        "sample_size": int(sample_size),
        "median_days": round(float(stay.median()), 1),
        "mean_days": round(float(stay.mean()), 1),
        "p25_days": round(float(stay.quantile(0.25)), 1),
        "p75_days": round(float(stay.quantile(0.75)), 1),
        "historical_under_or_equal_4_days_pct": round(pct_under_4, 1),
        "historical_between_4_and_10_days_pct": round(pct_4_to_10, 1),
        "historical_over_10_days_pct": round(pct_over_10, 1),
    }


def predict_animal(animal_type: str,

    sex_upon_intake: str = "",
    age_upon_intake: str = "",
    breed: str = "",
    color: str = "",
    intake_type: str = "Stray",
    intake_condition: str = "Normal",
    name: str = "",
    datetime: str = "",
    ) -> dict:
    """Predicts shelter length of stay for an animal.

    Args:
        animal_type: 'dog' or 'cat'
        sex_upon_intake: e.g. 'Intact Male', 'Spayed Female'
        age_upon_intake: e.g. '2 years', '5 months'
        breed: breed name e.g. 'Labrador Retriever Mix', 'Pit Bull'
        color: e.g. 'Black/White'
        intake_type: e.g. 'Stray', 'Owner Surrender'
        intake_condition: e.g. 'Normal', 'Injured'
        name: animal name or empty string if nameless
        datetime: date and time of intake
    """
    print(">>> [TOOL EXECUTION] predict_animal has been called!")
    model = load_model(animal_type)

    row = {
        "animal_type": animal_type,
        "sex_upon_intake": sex_upon_intake,
        "age_upon_intake": age_upon_intake,
        "breed": breed,
        "color": color,
        "intake_type": intake_type,
        "intake_condition": intake_condition,
        "name": name if name else None,
        "datetime": pd.to_datetime(datetime) if datetime else pd.Timestamp.now(),
        "found_location": "",
    }

    df = pd.DataFrame([row])

    df_fe = feature_engineering_days(df).copy()

    if df_fe.empty:
        raise ValueError("After feature engineering the df is empty")

    X = df_fe[model.feature_names_].copy()
    probs = model.predict_proba(X)[0]

    probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(model.classes_, probs)
    }

    high_risk_flag = False
    if animal_type == "dog":
        high_risk_flag = bool(probabilities[2] >= 0.230)   # Dog threshold
    elif animal_type == "cat":
        high_risk_flag = bool(probabilities[2] >= 0.453)   # Cat threshold
    return {
        "under_or_equal_4_days": probabilities[0],
        "between_4_and_10_days": probabilities[1],
        "over_10_days": probabilities[2],
        "high_risk_flag": high_risk_flag
    }
    
from catboost import Pool

def explain_animal(
    animal_type: str,
    sex_upon_intake: str = "",
    age_upon_intake: str = "",
    breed: str = "",
    color: str = "",
    intake_type: str = "Stray",
    intake_condition: str = "Normal",
    name: str = "",
    datetime: str = ""
) -> dict:
    """Explains which intake factors influence the predicted stay using SHAP.

    Args:
        animal_type: 'dog' or 'cat'
        sex_upon_intake: e.g. 'Intact Male'
        age_upon_intake: e.g. '7 years'
        breed: breed name
        color: color
        intake_type: e.g. 'Stray'
        intake_condition: condition
        name: animal name
        datetime: date of intake
    """

    print(">>> [TOOL EXECUTION] explain_animal has been called!", flush=True)

    model = load_model(animal_type)

    row = {
        "animal_type": animal_type,
        "sex_upon_intake": sex_upon_intake,
        "age_upon_intake": age_upon_intake,
        "breed": breed,
        "color": color,
        "intake_type": intake_type,
        "intake_condition": intake_condition,
        "name": name if name else None,
        "datetime": pd.to_datetime(datetime) if datetime else pd.Timestamp.now(),
        "found_location": ""
    }

    df = pd.DataFrame([row])
    df_fe = feature_engineering_days(df)
    X = df_fe[model.feature_names_].copy()

    # SHAP calculated directly by CatBoost
    pool = Pool(
        X,
        cat_features=model.get_cat_feature_indices()
    )

    shap_values = model.get_feature_importance(
        data=pool,
        type="ShapValues"
    )

    print(">>> SHAP calculated!", flush=True)

    class_names = {
        0: "under_or_equal_4_days",
        1: "between_4_and_10_days",
        2: "over_10_days",
    }

    explanations = {}

    for class_idx, cls in enumerate(model.classes_):

        # Multiclass SHAP shape:
        # (samples, classes, features + 1)
        # Last value is the expected value (baseline).
        values = shap_values[0, class_idx, :-1]

        explanation_df = pd.DataFrame({
            "feature": X.columns,
            "shap_value": values,
            "feature_value": X.iloc[0].values
        })

        explanation_df["importance"] = explanation_df["shap_value"].abs()

        top_features = (
            explanation_df
            .sort_values("importance", ascending=False)
            .head(5)
        )

        explanations[class_names[int(cls)]] = [
            {
                "feature": str(record["feature"]),
                "feature_value": (
                    None if pd.isna(record["feature_value"])
                    else str(record["feature_value"])
                ),
                "shap_value": float(record["shap_value"])
            }
            for record in top_features[
                ["feature", "feature_value", "shap_value"]
            ].to_dict(orient="records")
        ]

    return explanations