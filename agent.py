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

def predict_animal(animal_type: str,

    sex_upon_intake: str = "",
    age_upon_intake: str = "",
    breed: str = "",
    color: str = "",
    intake_type: str = "Stray",
    intake_condition: str = "Normal",
    name: str = "",
    datetime: str = "") -> dict:
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
        "found_location": ""
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
    
def explain_animal(animal_type: str,
    sex_upon_intake: str = "",
    age_upon_intake: str = "",
    breed: str = "",
    color: str = "",
    intake_type: str = "Stray",
    intake_condition: str = "Normal",
    name: str = "",
    datetime: str = "") -> dict:
    """Explains which intake factors most influence the long stay prediction using SHAP.

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
    print(">>> [TOOL EXECUTION] explain_animal has been called!")
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    class_names = {
        0: "under_or_equal_4_days", 
        1: "between_4_and_10_days",
        2: "over_10_days",
    }

    explanations = {}

    for class_idx, cls in enumerate(model.classes_):
        values = shap_values.values[0, :, class_idx]

        explanation_df = pd.DataFrame({
            "feature": X.columns, 
            "shap_value": values, 
            "feature_value": X.iloc[0].values
        })

        explanation_df["importance"] = explanation_df["shap_value"].abs()
        top_features = explanation_df.sort_values("importance", ascending=False).head(5)

        explanations[class_names[int(cls)]] = (
            top_features[["feature", "feature_value", "shap_value"]].to_dict(orient="records")
        )

    return explanations

def get_animal_info():
    print("animal-info")

def get_high_risk_animals():
    print("high-risk")

def shelter_statistics():
    print("shelter-statistics")




def run_shelter_agent(user_message: str):
    client = genai.Client()

    system_instruction = (
        "You are an AI assistant for animal shelter staff. "
        "When asked about expected length of stay for an incoming animal, use the `predict_animal` tool. "
        "When asked about reasons or explanations, use the `explain_animal` tool."
    )

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[predict_animal, explain_animal],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            temperature=0.0,
        ),
    )

    response = chat.send_message(user_message)

    if not response.candidates:
        return f"[Error] No candidates in response. PromptFeedback: {getattr(response, 'prompt_feedback', 'Empty')}"

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    parts = getattr(candidate.content, "parts", [])
    collected_texts = []
    for idx, part in enumerate(parts):
        if getattr(part, "text", None):
            collected_texts.append(part.text)

    if collected_texts:
        return "\n".join(collected_texts)

    return f"[Error] No response. Raw candidate:\n{candidate}"

def main():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing in .env")

    # Prosty test weryfikujący czy agent działa
    test_query = (
    "We just took in an intact male dog, about 7 years old, Pit Bull Mix. "
    "He was brought in as a stray with no collar or name, and his intake health condition is normal. "
    "How long is he expected to stay in the shelter, and what are the main reasons behind this prediction?"
)
    
    print("Question:", test_query)
    print("\nAgent Answer")
    print(run_shelter_agent(test_query))

if __name__ == "__main__":
    main()
