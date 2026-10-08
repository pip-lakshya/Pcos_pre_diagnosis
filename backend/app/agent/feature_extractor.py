"""Forced structured extraction of only explicitly reported raw PCOS features."""
import json
import re

from openai import OpenAI

from app.agent.session_store import BOOLEAN_FEATURES, INTEGER_FEATURES, RAW_FEATURES
from app.config import settings

_PROPERTIES = {}
for _name in RAW_FEATURES:
    if _name in BOOLEAN_FEATURES:
        _PROPERTIES[_name] = {
            "type": ["boolean", "null"],
            "description": "Use true for an explicit yes, false for an explicit no, or null if unstated.",
        }
    elif _name in INTEGER_FEATURES:
        _PROPERTIES[_name] = {"type": ["integer", "null"]}
    else:
        _PROPERTIES[_name] = {"type": ["number", "null"]}
    if _name == "hip_inch":
        _PROPERTIES[_name]["description"] = "Hip circumference in inches; convert an explicitly stated cm value by dividing by 2.54."
    elif _name == "waist_inch":
        _PROPERTIES[_name]["description"] = "Waist circumference in inches; convert an explicitly stated cm value by dividing by 2.54."
    elif _name == "weight_kg":
        _PROPERTIES[_name]["description"] = "Weight in kilograms; convert explicitly stated pounds to kg by dividing by 2.20462."
    elif _name == "height_cm":
        _PROPERTIES[_name]["description"] = "Height in centimeters; convert explicitly stated inches by multiplying by 2.54."

TOOL = {
    "type": "function",
    "function": {
        "name": "extract_pcos_features",
        "description": (
            "Extract only raw health feature values the user explicitly stated. "
            "Omit or use null for unstated fields. Convert explicitly stated units as described by each field. "
            "Never guess, derive health features, or add advice."
        ),
        "parameters": {
            "type": "object",
            "properties": _PROPERTIES,
            "additionalProperties": False,
        },
    },
}

SYSTEM = (
    "You are a structured data extraction agent, not a chatbot giving medical advice and not an agent "
    "diagnosing PCOS. This supports a screening estimate only, not a diagnosis. Extract only raw information the user explicitly stated into the "
    "extract_pcos_features schema. Never infer, assume, or guess a value. Unit conversion is allowed "
    "only when the source unit is explicitly stated. Do not calculate derived health features. "
    "Understand the user's messages in English, Hindi, Hinglish, and mixed-language speech/transcripts. "
    "For every boolean field, encode an explicit yes as JSON true and an explicit no as JSON false, "
    "never as the strings 'yes' or 'no'. Hindi/Hinglish yes examples include haan, ji, bilkul; "
    "no examples include nahi, nahin, nope, bilkul nahi. Interpret a short answer using recent turns. "
    "When the user explicitly corrects an earlier answer, or provides a feature that was missing earlier, "
    "return that value. Treat phrases such as 'my correct weight is 56', 'update my weight to 56', "
    "'mera sahi wazan 56 hai', and 'I forgot to mention my acne is yes' as explicit updates. "
    "Do not treat a labelled correction as confirmation of the whole summary. If the user gives a "
    "measurement in cm for hip or waist, convert it to "
    "inches for the corresponding schema field (1 inch = 2.54 cm); if the unit is unclear, do not guess. "
    "If the user says they do not know, prefer not to answer, or asks to skip a field, do not invent a value. "
    "If its meaning is unclear, leave that field null. "
    "Leave other unstated fields null or omit them. Do not calculate BMI or waist-to-hip ratio; "
    "those are calculated by the Python backend. Never calculate androgen_score, metabolic_score, "
    "physical_risk, or cycle_severity; those are Python-only derived annotations. Return tool output "
    "only; do not answer the user in prose."
)

_TRUE_WORDS = {"yes", "y", "yeah", "yep", "yup", "true", "1", "correct", "affirmative"}
_FALSE_WORDS = {"no", "n", "nope", "nah", "false", "0", "incorrect", "negative"}
_TRUE_WORDS.update({"haan", "han", "ji", "bilkul", "हाँ", "हां", "जी"})
_FALSE_WORDS.update({"nahi", "nahin", "nhi", "नहीं", "नही"})

_QUESTION_FEATURES = {
    "age": ("age", "how old", "years old"),
    "weight_kg": ("weight", "weigh"),
    "height_cm": ("height", "tall"),
    "cycle_irregular": ("irregular", "cycles", "periods regular"),
    "cycle_length_days": ("how many days", "days from", "cycle length", "days between"),
    "weight_gain": ("weight gain", "gained weight"),
    "hair_growth": ("hair growth", "coarse hair", "facial hair"),
    "skin_darkening": ("skin", "dark patches", "darkening"),
    "hair_loss": ("hair loss", "hair thinning"),
    "acne": ("acne", "pimples"),
    "fast_food": ("fast food", "takeaway"),
    "regular_exercise": ("exercise", "physical activity"),
    "hip_inch": ("hip measurement", "hips"),
    "waist_inch": ("waist measurement", "waist"),
}


def _normalize_boolean(value):
    """Convert clear boolean representations to bool; leave ambiguous values unanswered."""
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower().strip(".!? ")
        if normalized in _TRUE_WORDS:
            return True
        if normalized in _FALSE_WORDS:
            return False
    return None


def _recover_simple_answer(message: str, history: list) -> dict:
    """Recover a clear short answer using the immediately preceding assistant question."""
    previous_question = next(
        (turn.get("content", "").lower() for turn in reversed(history) if turn.get("role") == "assistant"),
        "",
    )
    matches = [
        (len(term), name)
        for name, terms in _QUESTION_FEATURES.items()
        for term in terms
        if term in previous_question
    ]
    target = max(matches)[1] if matches else None
    if not target:
        return {}

    value = message.strip()
    if target in BOOLEAN_FEATURES:
        normalized = _normalize_boolean(value)
        return {target: normalized} if normalized is not None else {}

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|lb|lbs|pounds?|cm|centimeters?|mm|in|inch|inches|days?|years?)?\s*[.!]?\s*",
        value, re.I,
    )
    if not match:
        return {}
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if target in {"hip_inch", "waist_inch"}:
        if unit in {"cm", "centimeter", "centimeters"}:
            number /= 2.54
        elif unit == "mm":
            number /= 25.4
    elif target == "height_cm" and unit in {"in", "inch", "inches"}:
        number *= 2.54
    elif target == "weight_kg" and unit in {"lb", "lbs", "pound", "pounds"}:
        number /= 2.20462
    if target in INTEGER_FEATURES:
        return {target: int(number)} if number.is_integer() else {}
    return {target: number}


def recover_labeled_numeric_update(message: str) -> dict:
    """Recover a clearly labelled numeric value even when it was not just asked."""
    aliases = {
        "age": ("age", "umar", "उम्र", "उमर"),
        "weight_kg": ("weight", "wazan", "wajan", "वज़न", "वजन"),
        "height_cm": ("height", "lambai", "लंबाई", "लम्बाई"),
        "cycle_length_days": ("cycle length", "period length", "cycle ke din", "cycle kitne din", "पीरियड साइकिल"),
    }
    unit_pattern = r"(kg|kilograms?|lb|lbs|pounds?|cm|cms|centimeters?|in|inches?)?"
    label_matches = []
    for feature, terms in aliases.items():
        for term in terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)" if term.isascii() else re.escape(term)
            label_matches.extend((match.start(), match.end(), feature) for match in re.finditer(pattern, message, re.I))
    detected_features = {item[2] for item in label_matches}
    # If the user updates several fields in one sentence, let the structured
    # extractor handle the whole message rather than consuming only one value.
    if len(detected_features) != 1:
        return {}
    feature = next(iter(detected_features))
    _, label_end, _ = max((item for item in label_matches if item[2] == feature), key=lambda item: item[0])
    number_matches = list(re.finditer(rf"(?<!\w)(\d+(?:\.\d+)?)\s*{unit_pattern}(?![\w])", message, re.I))
    if not number_matches:
        return {}
    after_label = [match for match in number_matches if match.start() >= label_end]
    if after_label:
        # Prefer the final value in “change weight from 55 to 56” and the value
        # immediately after the label in “my correct weight is 56”.
        to_value = [match for match in after_label if re.search(r"\bto\s*$", message[label_end:match.start()], re.I)]
        number_match = to_value[-1] if to_value else after_label[0]
    else:
        number_match = min(number_matches, key=lambda match: abs(match.end() - label_end))
    number = float(number_match.group(1))
    unit = (number_match.group(2) or "").lower()
    if feature == "weight_kg" and unit in {"lb", "lbs", "pound", "pounds"}:
        number /= 2.20462
    elif feature == "height_cm" and unit in {"in", "inch", "inches"}:
        number *= 2.54
    if feature in INTEGER_FEATURES:
        return {feature: int(number)} if number.is_integer() else {}
    return {feature: number}


def question_feature(history: list) -> str | None:
    """Find the feature in the latest assistant question for short answers/skips."""
    previous_question = next(
        (turn.get("content", "").lower() for turn in reversed(history) if turn.get("role") == "assistant"),
        "",
    )
    matches = [
        (len(term), name)
        for name, terms in _QUESTION_FEATURES.items()
        for term in terms
        if term in previous_question
    ]
    return max(matches)[1] if matches else None


def skip_targets(message: str, history: list) -> set[str]:
    """Prefer an explicitly named field; otherwise apply a short skip to the active question."""
    text = message.lower()
    aliases = {
        "age": ("age", "umar", "उम्र"),
        "weight_kg": ("weight", "wazan", "वज़न", "वजन"),
        "height_cm": ("height", "tall", "lambai", "लंबाई"),
        "cycle_irregular": ("cycle irregular", "irregular period", "periods", "mahwari", "पीरियड"),
        "cycle_length_days": ("cycle length", "days between", "kitne din"),
        "weight_gain": ("weight gain", "wazan badh", "वज़न बढ़"),
        "hair_growth": ("hair growth", "facial hair", "baal", "बाल"),
        "skin_darkening": ("skin dark", "dark patch", "त्वचा"),
        "hair_loss": ("hair loss", "hair fall", "baal jhad", "बाल झड़"),
        "acne": ("acne", "pimple", "मुंहासे"),
        "fast_food": ("fast food", "junk food"),
        "regular_exercise": ("exercise", "workout", "व्यायाम"),
        "hip_inch": ("hip", "hips", "कूल्हे"),
        "waist_inch": ("waist", "कमर"),
    }
    explicit = {name for name, terms in aliases.items() if any(term in text for term in terms)}
    if explicit:
        return explicit
    active = question_feature(history)
    return {active} if active else set()


def is_skip_answer(message: str) -> bool:
    text = message.lower().strip()
    return any(term in text for term in (
        "skip", "don't know", "dont know", "do not know", "not sure", "prefer not",
        "rather not", "i cannot say", "can't say", "pata nahi", "nahi pata", "नहीं पता",
        "छोड़", "छोड", "नहीं मालूम",
    ))


def _normalize_explicit_units(message: str, updates: dict) -> dict:
    """Apply conversions in Python when the message names a feature and its unit."""
    specs = {
        "hip_inch": (r"(?:\bhip(?:s)?\b|कूल्हे)", {"cm": 1 / 2.54, "centimeter": 1 / 2.54, "centimeters": 1 / 2.54, "cms": 1 / 2.54, "in": 1.0, "inch": 1.0, "inches": 1.0}),
        "waist_inch": (r"(?:\bwaist\b|कमर)", {"cm": 1 / 2.54, "centimeter": 1 / 2.54, "centimeters": 1 / 2.54, "cms": 1 / 2.54, "in": 1.0, "inch": 1.0, "inches": 1.0}),
        "height_cm": (r"(?:\bheight\b|लंबाई)", {"cm": 1.0, "centimeter": 1.0, "centimeters": 1.0, "cms": 1.0, "in": 2.54, "inch": 2.54, "inches": 2.54}),
        "weight_kg": (r"(?:\bweight\b|वज़न|वजन|wazan)", {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "lb": 1 / 2.20462, "lbs": 1 / 2.20462, "pound": 1 / 2.20462, "pounds": 1 / 2.20462}),
    }
    for feature, (label, unit_factors) in specs.items():
        match = re.search(
            rf"{label}[^0-9,;\n]{{0,40}}(\d+(?:\.\d+)?)\s*(kg|kilograms?|cm|cms|centimeters?|in|inches?|lb|lbs|pounds?)\b",
            message,
            re.I,
        )
        if not match or match.group(2).lower() not in unit_factors:
            continue
        value = float(match.group(1))
        updates[feature] = value * unit_factors[match.group(2).lower()]
    return updates


def extract(message: str, current: dict, history: list) -> dict:
    labeled_update = recover_labeled_numeric_update(message)
    if labeled_update:
        return labeled_update
    # The normal intake asks one field at a time. Handle concise answers locally
    # so a one-word yes/no or number does not wait on a remote model round trip.
    simple_answer = _recover_simple_answer(message, history)
    if simple_answer:
        return simple_answer

    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for structured feature extraction")

    client = OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=45.0,
        max_retries=0,
    )
    current_raw = {name: current[name] for name in RAW_FEATURES if name in current}
    recent_turns = history[-8:]
    prompt = (
        f"Recent conversation: {recent_turns}\n"
        f"Most recent assistant question: {next((turn.get('content', '') for turn in reversed(history) if turn.get('role') == 'assistant'), '')}\n"
        f"Already collected raw answers (do not overwrite unless explicitly corrected): {current_raw}\n"
        f"Latest user message: {message}"
    )
    response = client.chat.completions.create(
        model=settings.nvidia_model,
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tools=[TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_pcos_features"}},
        max_tokens=240,
    )

    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []
    matching_calls = [call for call in tool_calls if call.function.name == "extract_pcos_features"]
    if not matching_calls:
        return _recover_simple_answer(message, history)

    extracted = {}
    for call in matching_calls:
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict) or set(arguments) - set(RAW_FEATURES):
            raise ValueError("Feature extraction response contains an unknown or derived field")
        extracted.update(arguments)
    # Defensively normalize yes/no variants before strict session validation. A value we
    # cannot interpret is omitted, leaving that slot open for the normal follow-up flow.
    updates = {}
    for name, value in extracted.items():
        if value is None:
            continue
        if name in BOOLEAN_FEATURES:
            value = _normalize_boolean(value)
            if value is None:
                continue
        updates[name] = value
    return _normalize_explicit_units(message, updates)
