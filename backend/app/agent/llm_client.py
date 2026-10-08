import re

from openai import OpenAI

from app.agent.explainer import explain as fallback_explanation
from app.config import settings

_QUESTION_TEMPLATES = {
        "age": "How old are you?",
        "weight_kg": "What is your current weight in kilograms?",
        "height_cm": "What is your height in centimeters?",
        "cycle_irregular": "Are your menstrual cycles often irregular?",
        "cycle_length_days": "About how many days are there from the start of one period to the start of the next?",
        "weight_gain": "Have you noticed unexplained weight gain?",
        "hair_growth": "Have you noticed more coarse hair growth on your face or body?",
        "skin_darkening": "Have you noticed darker, velvety skin patches?",
        "hair_loss": "Have you noticed unusual hair thinning or loss?",
        "acne": "Have you had persistent acne?",
        "fast_food": "Do you often eat fast food?",
        "regular_exercise": "Do you exercise regularly?",
        "hip_inch": "Do you know your hip measurement in inches? (If you do not know write skip).",
        "waist_inch": "Do you know your waist measurement in inches? (If you do not know write skip).",
}

_HINDI_QUESTIONS = {
    "age": "Aapki umar kitni hai?",
    "weight_kg": "Aapka wazan kilogram mein kitna hai?",
    "height_cm": "Aapki height centimeter mein kitni hai?",
    "cycle_irregular": "Kya aapke periods aksar irregular hote hain?",
    "cycle_length_days": "Ek period ke pehle din se agle period ke pehle din tak lagbhag kitne din hote hain?",
    "weight_gain": "Kya haal mein aapka wazan badha hai?",
    "hair_growth": "Kya chehre ya badan par naye ya zyada baal aaye hain?",
    "skin_darkening": "Kya skin par naye dark patches ya kaalapan dikha hai?",
    "hair_loss": "Kya aapko baal jhadne ya patle hone ka anubhav hai?",
    "acne": "Kya aapko abhi acne ya pimples hote hain?",
    "fast_food": "Kya aap hafte mein kam se kam ek baar fast food khate hain?",
    "regular_exercise": "Kya aap regular exercise karte hain?",
    "hip_inch": "Agar aapko pata ho, hip measurement batayein. (Agar aap ko nahi pata to skip likh sakte hain).",
    "waist_inch": "Agar aapko pata ho, waist measurement batayein. (Agar aap ko nahi pata to skip likh sakte hain).",
}


def question_for_field(name: str) -> str:
    return _QUESTION_TEMPLATES.get(name, f"Could you share your {name.replace('_', ' ')}?")


def _uses_hinglish(history: list) -> bool:
    recent = " ".join(
        turn.get("content", "") for turn in history[-6:] if turn.get("role") == "user"
    ).lower()
    return bool(re.search(r"[\u0900-\u097f]|\b(haan|han|nahi|nahin|pata|umar|wazan|batao|hai|mera|meri|mujhe|kya|kyun|kaise|kab)\b", recent))


_FIELD_TERMS = {
    "age": ("age", "old", "years old"),
    "weight_kg": ("current weight", "your weight", "weight in kilograms", "how much do you weigh", "weight kg"),
    "height_cm": ("height", "how tall", "centimeters", "cm tall"),
    "cycle_irregular": ("irregular cycle", "irregular periods", "periods irregular", "cycles regular", "periods regular"),
    "cycle_length_days": ("cycle length", "days between", "days from", "periods apart", "between periods"),
    "weight_gain": ("weight gain", "gained weight"),
    "hair_growth": ("hair growth", "coarse hair", "facial hair"),
    "skin_darkening": ("skin darkening", "dark skin", "dark patches"),
    "hair_loss": ("hair loss", "hair thinning"),
    "acne": ("acne", "pimples"),
    "fast_food": ("fast food", "takeaway"),
    "regular_exercise": ("exercise", "physical activity"),
    "hip_inch": ("hip",),
    "waist_inch": ("waist",),
}


def _references_missing(reply: str, missing: list[str]) -> bool:
    text = re.sub(r"[^a-z0-9 ]", " ", reply.lower())
    return any(term in text for name in missing for term in _FIELD_TERMS.get(name, (name.replace("_", " "),)))


def fallback_question(missing: list[str], history: list | None = None) -> str:
    if not missing:
        return "What else would you like to share?"
    if history and _uses_hinglish(history):
        return _HINDI_QUESTIONS.get(missing[0], question_for_field(missing[0]))
    return question_for_field(missing[0])


def next_question(history: list, missing: list[str], confirmed: dict | None = None) -> str:
    # One short, deterministic question avoids a second network round trip on every turn.
    # The intro and completed result carry the screening disclaimer; intake questions do not repeat it.
    if not missing:
        return "What else would you like to share?"
    field = missing[0]
    if _uses_hinglish(history):
        return _HINDI_QUESTIONS.get(field, question_for_field(field))
    return question_for_field(field)


def agreement_note(agreement_score: float) -> str:
    if agreement_score >= 0.8:
        return "Our models strongly agree on this estimate."
    if agreement_score >= 0.6:
        return "Our models show some disagreement, so this estimate is less certain."
    return "Our models show noticeable disagreement, so this estimate is especially uncertain."


def explain_prediction(
    probability: float, risk_label: str, feature_importances: dict,
    agreement_score: float, history: list,
) -> str:
    """Generate a conversational result explanation; this call intentionally uses no tools."""
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for the final risk explanation")

    client = OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url, timeout=45.0, max_retries=0)
    messages = [
        {
            "role": "system",
            "content": (
                "You explain a PCOS screening model's result in warm, plain, empathetic language. "
                "Reply in the language the user has been using, including Hindi or Hinglish when appropriate. "
                "This is a screening estimate, not a diagnosis. Never diagnose PCOS, claim certainty, "
                "or invent causes. Explain that only a healthcare professional can evaluate the "
                "person and recommend discussing concerns with a doctor. Keep the response concise. "
                "Describe model agreement only with the supplied plain-language note. Never expose "
                "agreement numbers or member counts. Return natural conversational text only, not JSON or code."
            ),
        },
        *history[-6:],
        {
            "role": "user",
            "content": (
                f"The model ensemble's mean positive-class probability is {probability:.4f} "
                f"({probability * 100:.1f}%), and its screening label is {risk_label!r}. "
                f"Include this plain-language confidence note in your answer: {agreement_note(agreement_score)} "
                "Do not mention agreement numbers or how many models were used. "
                f"The model's feature_importances are {list(feature_importances.items())[:5]}. "
                "Explain the screening result in plain language. You may describe these as broad "
                "model input importance, not as causes or proof about this person. Do not change "
                "the model probability or invent another risk number."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=settings.nvidia_model,
        temperature=0.5,
        messages=messages,
        max_tokens=220,
    )
    note = agreement_note(agreement_score)
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    text = getattr(message, "content", None)
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in text
        )
    if not isinstance(text, str) or not text.strip():
        # Complete and persist the screening even when the optional language
        # generation call returns no user-facing text. The model's prediction
        # remains the sole source of the estimate.
        return f"{fallback_explanation(probability, risk_label)} Please consult a doctor about personal concerns. {note}"

    answer = text.strip()
    if note.lower() not in answer.lower():
        answer = f"{answer} {note}"
    if "doctor" not in answer.lower() and "healthcare professional" not in answer.lower():
        answer += " Please discuss personal concerns with a doctor."
    return answer
