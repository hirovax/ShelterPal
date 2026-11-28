import os

from google import genai
from google.genai import types

from pathlib import Path
from agent_tools import predict_animal, query_history, explain_animal

from dotenv import load_dotenv

MODELS_DIR = Path(__file__).resolve().parent / "models"
RAW_DIR = Path(__file__).resolve().parent / "data" / "austin" / "raw"

def run_shelter_agent(user_message: str):
    client = genai.Client()

    system_instruction = """
    You are ShelterPal, an AI assistant for animal shelter staff.

    Your responsibilities:
    - Predict animal shelter stay probabilities using predict_animal.
    - Explain model predictions using explain_animal.
    - Compare animals with historical records using query_history.

    When analyzing an animal:
    1. Use predict_animal to estimate the length of stay.
    2. Use explain_animal to identify the main contributing factors.
    3. Use query_history when historical comparison is relevant.

    If predict_animal returns high_risk_flag=True:
    - Automatically suggest 2-3 personalized actions that
    could help improve the animal's adoption chances.
    - Consider the animal's characteristics, SHAP explanations,
    and available historical information.
    - Independently decide which actions are most appropriate.
    - Explain why each recommendation could be useful.

    If high_risk_flag=False:
    - Focus on the prediction and explanation.
    - Provide recommendations only if explicitly requested.

    Important:
    - All adoption recommendations must be ethical, realistic,
    and focused on the animal's welfare.

    - Base recommendations on the actual CatBoost predictions
    and SHAP explanations, not on unsupported assumptions.

    - Use SHAP to identify potentially relevant factors
    and consider whether they suggest practical actions.

    - Prioritize actionable factors when making recommendations,
    but do not ignore important non-actionable factors.

    - You may suggest changes related to modifiable features,
    such as assigning a name to an unnamed animal or
    considering a shorter or longer name when appropriate.

    - However, never recommend changing an animal's name,
    name length, or any other attribute solely to manipulate
    the model's prediction.

    - SHAP values describe contributions to model predictions.
    They do not establish that changing a feature will
    improve adoption outcomes.

    - Never suggest falsifying or manipulating animal records,
    including age, breed, sex, health condition, or intake history.

    - Never recommend unnecessary medical procedures
    solely to improve predicted adoption chances.

    - Do not invent an animal's temperament, behavior,
    medical history, or compatibility with adopters.

    - Clearly distinguish confirmed animal information
    from assumptions and possible recommendations.

    - Present all proposed actions as suggestions,
    not as guaranteed solutions.

    - Never guarantee that any recommendation will
    reduce shelter stay or lead to adoption.

    - Explain briefly why each recommendation may be useful
    and, where relevant, which SHAP factors informed it.

    - Remember that predicted shelter length of stay
    is not equivalent to predicted time until adoption.

    - Always prioritize responsible adoption and animal welfare
    over simply reducing the predicted shelter stay.

    - For animals with intake_condition="Sick" or "Nursing",
    consider that animals will need necessary care or specialized support.

    - Historical data indicates that transfers are common
    among Sick and Nursing animals, but check it with shap and query history.

    - Do not interpret a transfer as a completed adoption.

    - When suggesting actions for Sick or Nursing animals,
    prioritize animal welfare, appropriate care,
    and responsible placement over reducing shelter stay.

    - Never recommend transfer solely to shorten
    the predicted length of stay.
    """

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[predict_animal, explain_animal,query_history],
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
        "We just took in an intact male dog, approximately 2 years old, "
        "American Pit Bull Terrier Mix. "
        "He was brought in as a stray, has no name, "
        "and his intake health condition is normal. "
        "Set the risk flag to False"
        "How long is he expected to stay in the shelter, "
        "and what are the main reasons behind this prediction? "
        "How does his situation compare to similar dogs in our shelter? "
    )
    
    print("Question:", test_query)
    print("\nAgent Answer")
    print(run_shelter_agent(test_query))

if __name__ == "__main__":
    main()
