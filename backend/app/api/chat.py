import re
from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.feature_extractor import (
    classify_boolean_reply,
    extract,
    is_skip_answer,
    recover_labeled_numeric_update,
    skip_targets,
)
from app.agent.llm_client import explain_prediction, fallback_question, next_question
from app.agent.research_provider import KnowledgeBaseProvider
from app.agent.session_store import BOOLEAN_FEATURES, INTAKE_ORDER, OPTIONAL, derive_features, get_session, merge_features, missing
from app.auth.security import get_current_user
from app.db.database import get_db
from app.db.models import Screening, User
from app.ml.predictor import predict
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()
MAX_INTAKE_TURNS = 15
_education_provider = KnowledgeBaseProvider()


def _chat_response(session, request: ChatRequest, **fields) -> ChatResponse:
    response = ChatResponse(**fields)
    if request.message_id:
        session.response_cache[request.message_id] = response.model_dump(mode="json")
        while len(session.response_cache) > 50:
            session.response_cache.pop(next(iter(session.response_cache)))
    return response


def _next_session_question(session) -> str:
    outstanding = missing(session)
    session.active_question_id = outstanding[0] if outstanding else None
    return next_question(session.history, outstanding, session.values)


def _is_repeat_request(message: str) -> bool:
    return bool(re.search(r"\b(repeat|say that again|ask me again|repeat the question)\b|(?:phir se pucho|dobara pucho|फिर से पूछो)", message, re.I))


def _is_answer_reminder(message: str) -> bool:
    return bool(re.search(r"\b(i said|i already said|i told you|i have already answered|i already answered)\b|"
                          r"(maine kaha|maine pehle hi bataya|maine jawab diya)", message, re.I))


def _question_for_active(session) -> str:
    if session.active_question_id:
        return next_question(session.history, [session.active_question_id], session.values)
    return _next_session_question(session)


def _answer_during_screening(message: str, session) -> str:
    question = _question_for_active(session)
    if re.search(r"\bwhy\b|kyun|क्यों", message, re.I):
        answer = "I ask this as one self-reported screening question. It does not diagnose PCOS; you can answer yes or no, say you don’t know, or skip an optional measurement."
    else:
        try:
            answer = _education_provider.answer(message, {"recent_conversation": session.history[-6:]})
        except Exception:
            snippets = _education_provider.retrieve(message)
            answer = " ".join(item["text"] for item in snippets[:2]) if snippets else "I can help with general PCOS information, but I don’t have enough curated information to answer that reliably. Please consult a healthcare professional for personal concerns."
    return f"{answer}\n\nWe can continue with: {question}"

_SUMMARY_INTENT = re.compile(
    r"\b(show|give|summari[sz]e|display|tell me|what have you recorded|what did i provide)\b.{0,50}\b(summary|answers|details|screening)\b|"
    r"\b(summary|answers|details)\b.{0,30}\b(show|please|my|current|updated)\b|"
    r"\bwhat have you recorded\b|\bwhat did i answer\b|\bwhat have i answered\b|\bwhat did i provide\b|"
    r"(meri summary|mera summary|summary dikhao|mere jawab|kya record kiya|meri details)", re.I
)
_FIELD_ALIASES = {
    "age": ("age", "umar", "उम्र"), "weight_kg": ("weight", "wazan", "wajan", "वजन", "वज़न"),
    "height_cm": ("height", "lambai", "लंबाई"), "cycle_irregular": ("irregular period", "irregular cycles", "cycle regularity", "regular cycles", "periods", "period"),
    "cycle_length_days": ("cycle length", "period length", "days between periods"),
    "weight_gain": ("weight gain", "wazan gain"), "hair_growth": ("hair growth", "facial hair", "body hair"),
    "skin_darkening": ("skin darkening", "dark skin", "dark patches"), "hair_loss": ("hair loss", "hair fall"),
    "acne": ("acne", "pimple"), "fast_food": ("fast food", "junk food"),
    "regular_exercise": ("exercise", "workout"), "hip_inch": ("hip measurement", "hips", "hip", "कूल्हे"),
    "waist_inch": ("waist measurement", "waist", "कमर"),
}


def _is_summary_request(message: str) -> bool:
    return bool(_SUMMARY_INTENT.search(message))


def _field_requested(message: str) -> str | None:
    text = message.lower()
    found = {key for key, aliases in _FIELD_ALIASES.items() if any(re.search(
        rf"(?<!\w){re.escape(alias)}(?!\w)" if alias.isascii() else re.escape(alias), text, re.I
    ) for alias in aliases)}
    # A query about cycles can refer to regularity or length; ask rather than guess.
    return next(iter(found)) if len(found) == 1 else None


def _is_personal_lookup(message: str) -> bool:
    text = message.lower()
    return bool(_field_requested(message) and re.search(
        r"\b(what|which|did i|have i|what was|what is|tell me|recorded|answer|report|provide|missing|not provided)\b|"
        r"(kya|maine|bataya|diya|mera|meri|kitna|kitni)", text, re.I
    )) or bool(re.search(r"(which answers|what answers).{0,20}(missing|not provided)|kya.{0,20}(nahi bataya|missing)|"
                         r"(current|my|mera|meri).{0,25}(estimate|score|risk percentage)|(estimate|score|risk).{0,20}(current|my|mera|meri)", text, re.I))


def _explicit_correction(message: str) -> tuple[dict, str | None, str | None]:
    """Return (updates, clarification, short change label) for unambiguous user corrections."""
    if _is_affirmative_confirmation(message):
        return {}, None, None
    if re.fullmatch(r"\s*(?:yes|yeah|yep|yup|no|nope|nah|haan|han|ji|nahi|nahin|nhi|हाँ|हां|नहीं|नही)[.! ]*", message, re.I):
        return {}, None, None
    numeric = recover_labeled_numeric_update(message)
    if numeric:
        return numeric, None, next(iter(numeric))
    field = _field_requested(message)
    correction_marker = re.search(
        r"\b(correct|correction|update|change|actually|mistake|wrong|incorrect|should be|instead|sahi|galat|badal)\b|"
        r"(sahi|galat|badalna|sahi hai|गलत|सही)", message, re.I
    )
    if not correction_marker and not re.search(r"\b(no|nope|nah|nahi|nahin|हाँ|हां|yes)\b", message, re.I):
        return {}, None, None
    if field is None:
        return {}, "Which answer would you like to correct, and what is the correct value?", None
    if field not in BOOLEAN_FEATURES:
        return {}, "Please give the corrected value with the field name and unit, for example: ‘my waist is 30 inches’.", None
    if re.search(r"\b(sometimes|occasionally|not always)\b", message, re.I):
        return {}, "I can only store this screening answer as yes or no. Should I record it as yes or no?", None
    yes = bool(re.search(r"\b(yes|yeah|haan|han|true|correct|regular|occasionally|sometimes)\b|(?:हाँ|हां|सही)", message, re.I))
    no = bool(re.search(r"\b(no|nope|nah|nahi|nahin|false|irregular)\b|(?:नहीं|नही)", message, re.I))
    if yes == no:
        return {}, "For that yes/no answer, please say yes or no. If you mean ‘sometimes’, tell me whether to record it as yes or no.", None
    return {field: yes}, None, field


def _summary_text(values: dict, skipped: set[str], pending: list[str], screening: Screening | None, hinglish: bool = False) -> str:
    labels = {
        "age": ("Age", "Umar"), "weight_kg": ("Weight", "Wazan"), "height_cm": ("Height", "Height"),
        "cycle_irregular": ("Cycle regularity", "Periods ki regularity"), "cycle_length_days": ("Cycle length", "Cycle length"),
        "weight_gain": ("Weight gain", "Weight gain"), "hair_growth": ("Increased hair growth", "Zyada hair growth"),
        "skin_darkening": ("Skin darkening", "Skin darkening"), "hair_loss": ("Hair loss", "Hair loss"),
        "acne": ("Acne", "Acne"), "fast_food": ("Frequent fast food", "Fast food"),
        "regular_exercise": ("Regular exercise", "Regular exercise"), "hip_inch": ("Hip measurement", "Hip measurement"),
        "waist_inch": ("Waist measurement", "Waist measurement"),
    }
    parts = []
    for key in INTAKE_ORDER:
        label = labels[key][1 if hinglish else 0]
        if key not in values:
            state = "Skipped" if key in skipped else "Not provided"
            parts.append(f"{label}: {state}")
            continue
        value = values[key]
        if key == "cycle_irregular": value = ("irregular" if value else "regular")
        elif key in BOOLEAN_FEATURES:
            value = ("haan" if value else "nahi") if hinglish else ("yes" if value else "no")
        elif key == "weight_kg": value = f"{value:g} kg"
        elif key == "height_cm": value = f"{value:g} cm"
        elif key in {"hip_inch", "waist_inch"}: value = f"{value:g} inches"
        elif key in {"age", "cycle_length_days"}: value = f"{value} {'saal' if key == 'age' and hinglish else 'years' if key == 'age' else 'din' if hinglish else 'days'}"
        parts.append(f"{label}: {value}")
    for key, label in (("bmi", "Calculated BMI"), ("waist_hip_ratio", "Calculated waist-to-hip ratio")):
        if key in values: parts.append(f"{label}: {values[key]:g} (calculated from recorded measurements)")
    result = "Not calculated yet for this screening."
    if screening is not None:
        result = f"{screening.risk_probability * 100:.0f}% · {screening.risk_label} range (saved screening estimate; not a diagnosis)."
    if not values:
        intro = "Abhi tak koi jawab record nahi hua." if hinglish else "There is not enough information for a complete summary yet."
    else:
        intro = "Aapki current screening summary:" if hinglish else "Your current screening summary:"
    missing_text = ", ".join(labels[name][1 if hinglish else 0] for name in pending) or "None"
    tail = (f"\n\nScreening estimate: {result}\nAbhi baaki: {missing_text}. Aap jawab de sakte hain ya optional measurements skip kar sakte hain." if hinglish else
            f"\n\nScreening estimate: {result}\nStill missing: {missing_text}. You can provide these answers or skip optional measurements.")
    return intro + "\n" + "\n".join(f"• {part}" for part in parts) + tail


def _lookup_reply(message: str, session, screening: Screening | None) -> str | None:
    text = message.lower()
    hinglish = _uses_hinglish(session.history)
    if re.search(r"(which answers|what answers).{0,20}(missing|not provided)|kya.{0,20}(nahi bataya|missing)", text, re.I):
        labels = {name: aliases[0] for name, aliases in _FIELD_ALIASES.items()}
        absent = [labels[name] for name in missing(session)]
        return ("Abhi yeh details nahi di gayi hain: " if hinglish else "Not provided yet: ") + (", ".join(absent) or "none")
    field = _field_requested(message)
    if field is None:
        if re.search(r"(estimate|score|risk).{0,25}(current|my|mera|meri)?|(current|my|mera|meri).{0,25}(estimate|score|risk)", text):
            if screening:
                return f"This screening’s saved estimate is {screening.risk_probability * 100:.0f}% ({screening.risk_label} range). It is not a diagnosis."
            return "There is no screening estimate for this active conversation yet."
        return None
    if field not in session.values:
        label = _FIELD_ALIASES[field][0]
        return (f"Is screening ke liye {label} record nahi hai. Aap ab bata sakte hain." if hinglish else
                f"I don’t have a {label} recorded for this screening. You can provide it now or skip it if optional.")
    value = session.values[field]
    if field == "cycle_irregular": value = ("irregular" if value else "regular")
    elif field in BOOLEAN_FEATURES: value = ("haan" if value else "nahi") if hinglish else ("yes" if value else "no")
    elif field == "weight_kg": value = f"{value:g} kg"
    elif field == "height_cm": value = f"{value:g} cm"
    elif field in {"hip_inch", "waist_inch"}: value = f"{value:g} inches"
    elif field in {"age", "cycle_length_days"}: value = str(value)
    label = _FIELD_ALIASES[field][0]
    prefix = "Aapke record mein" if hinglish else "Your recorded"
    return f"{prefix} {label} {'hai' if hinglish else 'is'} {value}."


def _looks_like_question(message: str) -> bool:
    text = message.strip().lower()
    if "?" in text:
        return True
    return bool(re.search(
        r"\b(what|why|how|when|where|who|which|can you|could you|tell me|explain|"
        r"kya|kyun|kaise|kab|kaunsa|kaunsi|batao|samjhao|hota hai|hoti hai|क्या|क्यों|कैसे|कब|बताइए|समझाइए)",
        text,
    ))


def _is_start_confirmation(message: str) -> bool:
    text = re.sub(r"[^\w']+", " ", message.lower(), flags=re.UNICODE).strip()
    start_words = r"(?:start|begin|shuru(?:\s+karo)?|chalu(?:\s+karo)?|go ahead|kar do|शुरू(?:\s+करो)?)"
    affirmative = r"(?:yes|yeah|yep|sure|okay|ok|haan|han|ji|bilkul|हाँ|हां|जी)"
    return bool(re.fullmatch(rf"(?:(?:{affirmative})\s+)*(?:please\s+)?{start_words}(?:\s+(?:please|now|karo|kar do))?", text)) \
        or bool(re.fullmatch(affirmative, text))


def _is_declining_start(message: str) -> bool:
    return bool(re.fullmatch(
        r"\s*(?:no|nope|nah|not now|maybe later|nahi|nahin|nhi|नहीं|अभी नहीं)[.! ]*",
        message.lower(),
    ))


def _is_restart_intent(message: str) -> bool:
    raw_text = message.lower()
    if any(phrase in raw_text for phrase in ("फिर से शुरू", "नया टेस्ट", "नई चैट", "नया स्क्रीनिंग")):
        return True
    text = re.sub(r"[^\w]+", " ", message.lower(), flags=re.UNICODE).strip()
    patterns = (
        r"\brestart(?: the)?(?: pcos)?(?: screening| test| chat)?\b",
        r"\bstart (?:the )?(?:screening|test|chat) again\b",
        r"\bstart over\b", r"\bstart from (?:the )?beginning\b",
        r"\bnew (?:chat|conversation|test|screening)\b",
        r"\b(?:another|new) pcos test\b", r"\btake (?:the )?(?:test|screening) again\b",
        r"\btest again\b", r"\bphir(?: se)? shuru(?: karo)?\b",
        r"\bdobara shuru(?: karo)?\b", r"\bnaya test\b", r"\bnayi chat\b",
        r"\bफिर से शुरू(?: करो)?\b", r"\bनया टेस्ट\b", r"\bनई चैट\b",
    )
    return len(text) <= 80 and any(re.search(pattern, text) for pattern in patterns)


def _is_correction_message(message: str) -> bool:
    if recover_labeled_numeric_update(message):
        return True
    text = message.lower()
    labels = (
        "age", "umar", "उम्र", "weight", "wazan", "wajan", "वज़न", "वजन",
        "height", "lambai", "लंबाई", "cycle", "period", "पीरियड", "hip", "कूल्हे",
        "waist", "कमर", "acne", "pimple", "hair", "baal", "बाल", "exercise",
        "skin", "फास्ट फूड",
    )
    correction_words = (
        "correct", "correction", "update", "change", "actually", "i meant", "i forgot",
        "forgot to mention", "sahi", "sahi value", "badal", "galat", "गलत", "सही",
    )
    has_label = any(label in text for label in labels)
    has_correction_marker = any(word in text for word in correction_words)
    has_explicit_boolean = bool(re.search(
        r"\b(?:yes|no|true|false|haan|han|nahi|nahin|nhi)\b|(?:हाँ|हां|नहीं|नही)", text
    ))
    return has_label and (has_correction_marker or has_explicit_boolean)


def _answer_question_then_offer_start(message: str, history: list) -> str:
    try:
        answer = _education_provider.answer(message, {"recent_conversation": history[-6:]})
    except Exception:
        # Keep PCOS questions usable if the optional generation service is down.
        excerpts = _education_provider.retrieve(message)
        if excerpts:
            answer = " ".join(item["text"] for item in excerpts[:2])
            answer += "\n\nThis is general screening education, not a diagnosis; please consult a doctor about personal concerns."
        else:
            answer = "I don’t have enough information in my curated PCOS materials to answer that reliably. Please consult a healthcare professional about your specific situation."
    hinglish = _uses_hinglish(history)
    offer = (
        "Kya main aapka screening ab shuru karun? Aap haan keh sakte hain, ya PCOS ke baare mein aur pooch sakte hain."
        if hinglish else
        "Would you like me to start your screening now? You can say yes, or ask another PCOS question."
    )
    return f"{answer}\n\n{offer}"


def _is_affirmative_confirmation(message: str) -> bool:
    if _is_correction_message(message):
        return False
    text = re.sub(r"[^\w']+", " ", message.lower(), flags=re.UNICODE).strip()
    if re.search(r"\b(no|nope|nahi|nahin|not|but|except|however|actually|instead|change|correction|गलत|नहीं)\b", text):
        return False
    return bool(re.search(
        r"\b(yes|yeah|yep|yup|haan|han|ji|bilkul|ok|okay|sure|correct|right|accurate|confirm|confirmed|agree|good|looks good|sounds right|think so|हाँ|हां|जी|सही)\b",
        text,
    ))


def _uses_hinglish(history: list) -> bool:
    recent = " ".join(
        turn.get("content", "") for turn in history[-8:] if turn.get("role") == "user"
    ).lower()
    return bool(re.search(r"[\u0900-\u097f]|\b(haan|han|nahi|nahin|pata|umar|wazan|batao|hai|mera|meri|mujhe|kya|kyun|kaise|kab)\b", recent))


def _confirmation_summary(values: dict, skipped: set[str] | None = None, hinglish: bool = False) -> str:
    skipped = skipped or set()
    cycles = "irregular" if values.get("cycle_irregular") else "regular"
    yes_no = lambda name: "yes" if values.get(name) else "no"
    details = [
        f"age: {values['age'] if 'age' in values else 'not provided'}",
        f"cycle regularity: {cycles if 'cycle_irregular' in values else 'not provided'}",
        f"cycle length: {values['cycle_length_days'] if 'cycle_length_days' in values else 'not provided'} days",
        f"weight: {values['weight_kg']:g} kg" if 'weight_kg' in values else "weight: not provided",
        f"height: {values['height_cm']:g} cm" if 'height_cm' in values else "height: not provided",
        f"weight gain: {yes_no('weight_gain') if 'weight_gain' in values else 'not provided'}",
        f"increased hair growth: {yes_no('hair_growth') if 'hair_growth' in values else 'not provided'}",
        f"skin darkening: {yes_no('skin_darkening') if 'skin_darkening' in values else 'not provided'}",
        f"hair loss: {yes_no('hair_loss') if 'hair_loss' in values else 'not provided'}",
        f"acne: {yes_no('acne') if 'acne' in values else 'not provided'}",
        f"frequent fast food: {yes_no('fast_food') if 'fast_food' in values else 'not provided'}",
        f"regular exercise: {yes_no('regular_exercise') if 'regular_exercise' in values else 'not provided'}",
    ]
    for key, label in (("hip_inch", "hip"), ("waist_inch", "waist")):
        if key in values:
            details.append(f"{label} measurement: {values[key]:g} inches")
    if skipped:
        details.append("skipped: " + ", ".join(sorted(skipped)))
    if hinglish:
        yn_hinglish = lambda name: "haan" if values.get(name) else "nahi"
        details = [
            f"aapki umar: {values['age']} saal" if "age" in values else "umar: nahi batayi",
            f"periods: {'irregular' if values.get('cycle_irregular') else 'regular'}" if "cycle_irregular" in values else "periods: nahi bataye",
            f"cycle length: {values['cycle_length_days']} din" if "cycle_length_days" in values else "cycle length: nahi batayi",
            f"wazan: {values['weight_kg']:g} kg" if "weight_kg" in values else "wazan: nahi bataya",
            f"height: {values['height_cm']:g} cm" if "height_cm" in values else "height: nahi batayi",
            f"weight gain: {yn_hinglish('weight_gain')}" if "weight_gain" in values else "weight gain: nahi bataya",
            f"zyada facial/body hair: {yn_hinglish('hair_growth')}" if "hair_growth" in values else "hair growth: nahi bataya",
            f"skin darkening: {yn_hinglish('skin_darkening')}" if "skin_darkening" in values else "skin darkening: nahi bataya",
            f"hair loss: {yn_hinglish('hair_loss')}" if "hair_loss" in values else "hair loss: nahi bataya",
            f"acne: {yn_hinglish('acne')}" if "acne" in values else "acne: nahi bataya",
            f"fast food: {yn_hinglish('fast_food')}" if "fast_food" in values else "fast food: nahi bataya",
            f"regular exercise: {yn_hinglish('regular_exercise')}" if "regular_exercise" in values else "exercise: nahi bataya",
        ]
        for key, label in (("hip_inch", "hip"), ("waist_inch", "waist")):
            if key in values:
                details.append(f"{label}: {values[key]:g} inches")
        if skipped:
            details.append("skip kiya: " + ", ".join(sorted(skipped)))
    return (
        ("Aapne jo details batayi hain, please check kar lein: " if hinglish else
         "Before I calculate your screening estimate, please check that I understood you: ")
        + "; ".join(details)
        + (". Kya yeh sahi hai? Haan kehkar confirm karein, ya koi detail update karne ke liye jaise ‘mera sahi wazan 56 hai’ ya ‘update my weight to 56 kg’ likhein. Agar pehle chhuti hui detail batani ho, woh bhi ab bata sakte hain." if hinglish else
           ". Is this accurate? Say yes to confirm, or update a detail—for example, ‘my correct weight is 56’ or ‘update my weight to 56 kg’. You can also add a detail you left out earlier.")
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    restart_requested = _is_restart_intent(request.message)
    session_id, session = get_session(None if restart_requested else request.session_id, user.id)
    if request.message_id and request.message_id in session.response_cache:
        return ChatResponse.model_validate(session.response_cache[request.message_id])
    if restart_requested:
        session.screening_started = True
        session.history.append({"role": "user", "content": request.message})
        reply = _next_session_question(session)
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request,
            session_id=session_id, reply=reply, collected=session.values,
            missing=missing(session), complete=False,
        )
    session.history.append({"role": "user", "content": request.message})
    screening = None
    if session.screening_id is not None:
        screening = db.query(Screening).filter(
            Screening.id == session.screening_id, Screening.user_id == user.id,
            Screening.completed.is_(True),
        ).first()

    if _is_summary_request(request.message):
        reply = _summary_text(session.values, session.skipped, missing(session), screening, _uses_hinglish(session.history))
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request,
            session_id=session_id, reply=reply, collected=session.values, missing=missing(session),
            complete=session.screening_id is not None, screening_id=session.screening_id,
            prediction=None, awaiting_confirmation=session.awaiting_confirmation, message_type="summary",
        )

    # Repair stale pointers from older in-memory sessions; never let a question
    # already present in the canonical feature state remain active.
    outstanding_before = missing(session)
    if session.screening_started and not session.awaiting_confirmation:
        if session.active_question_id not in outstanding_before:
            session.active_question_id = outstanding_before[0] if outstanding_before else None

    if _is_personal_lookup(request.message):
        reply = _lookup_reply(request.message, session, screening) or "Please tell me which recorded detail you want to check."
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request,
            session_id=session_id, reply=reply, collected=session.values, missing=missing(session),
            complete=session.screening_id is not None, screening_id=session.screening_id,
            awaiting_confirmation=session.awaiting_confirmation,
        )

    correction_updates, clarification, corrected_field = _explicit_correction(request.message)
    correction_context = session.awaiting_confirmation or session.screening_id is not None or _is_correction_message(request.message) or bool(re.search(
        r"\b(change|correct|update|previous answer|earlier answer|mistake|incorrect|wrong)\b|"
        r"(jawab.{0,20}(badal|galat|sahi)|गलत|सही)", request.message, re.I))
    if correction_context and (correction_updates or clarification):
        if clarification:
            session.history.append({"role": "assistant", "content": clarification})
            return _chat_response(session, request, session_id=session_id, reply=clarification, collected=session.values,
                missing=missing(session), complete=session.screening_id is not None,
                screening_id=session.screening_id, awaiting_confirmation=session.awaiting_confirmation)
        candidate = deepcopy(session)
        try:
            changed = merge_features(candidate, correction_updates)
            derive_features(candidate)
        except (TypeError, ValueError) as error:
            reply = f"I couldn’t apply that correction, and your saved answer is unchanged: {error}. Please provide a valid value with its unit."
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request, session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=session.screening_id is not None,
                screening_id=session.screening_id, awaiting_confirmation=session.awaiting_confirmation)
        if not changed:
            reply = "That answer already has this value.\n\n" + _summary_text(
                session.values, session.skipped, missing(session), screening, _uses_hinglish(session.history))
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request, session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=session.screening_id is not None,
                screening_id=session.screening_id, awaiting_confirmation=session.awaiting_confirmation,
                message_type="summary")
        if session.screening_id is not None:
            # Keep completed history immutable: successful corrections create a new
            # completed snapshot only after recalculation and DB commit both succeed.
            payload = {**candidate.values}
            for feature in OPTIONAL:
                payload.setdefault(feature, 0.0)
            from app.ml.predictor import load_assets
            _, model_features = load_assets()
            model_payload = {name: payload[name] for name in model_features}
            try:
                output = predict(model_payload)
                corrected_screening = Screening(
                    user_id=user.id,
                    feature_state_json={"features": payload, "feature_importances": output.feature_importances,
                        "agreement_score": output.agreement_score, "model_probabilities": output.model_probabilities},
                    risk_probability=output.probability, risk_label=output.risk_label, completed=True,
                )
                db.add(corrected_screening)
                db.commit()
                db.refresh(corrected_screening)
            except Exception as error:
                db.rollback()
                reply = "I couldn’t save and recalculate that correction, so your previous answer and estimate remain unchanged. Please try again."
                session.history.append({"role": "assistant", "content": reply})
                raise HTTPException(status_code=503, detail=reply) from error
            candidate.screening_id = corrected_screening.id
            candidate.history = session.history
            session.values = candidate.values
            session.skipped = candidate.skipped
            session.screening_id = corrected_screening.id
            screening = corrected_screening
            change_text = f"Updated {corrected_field.replace('_', ' ')}; the estimate has been recalculated."
            prediction = output
            is_complete = True
        else:
            session.values = candidate.values
            session.skipped = candidate.skipped
            if corrected_field in session.skipped:
                session.skipped.discard(corrected_field)
            change_text = f"Updated {corrected_field.replace('_', ' ')}."
            prediction = None
            is_complete = False
        reply = change_text + "\n\n" + _summary_text(
            session.values, session.skipped, missing(session), screening, _uses_hinglish(session.history))
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request, session_id=session_id, reply=reply, collected=session.values,
            missing=missing(session), complete=is_complete, prediction=prediction,
            screening_id=session.screening_id, awaiting_confirmation=session.awaiting_confirmation,
            message_type="summary")

    if session.screening_id is not None:
        raise HTTPException(status_code=409, detail="This screening is complete; start a new session to add or change other information.")

    if session.screening_started and _is_repeat_request(request.message):
        reply = _question_for_active(session)
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request, session_id=session_id, reply=reply,
            collected=session.values, missing=missing(session), complete=False,
            awaiting_confirmation=session.awaiting_confirmation)

    if session.screening_started and _is_answer_reminder(request.message) and classify_boolean_reply(request.message) is None:
        previous = session.last_answered_question_id
        if previous and previous in session.values:
            prompt = _question_for_active(session)
            reply = f"I have your previous answer recorded. Your current unanswered question is: {prompt}"
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request, session_id=session_id, reply=reply,
                collected=session.values, missing=missing(session), complete=False,
                awaiting_confirmation=session.awaiting_confirmation)

    if not session.screening_started:
        if _looks_like_question(request.message):
            reply = _answer_question_then_offer_start(request.message, session.history)
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
        if _is_declining_start(request.message):
            reply = "Theek hai, screening abhi shuru nahi karunga. Jab aap ready hon, bas ‘start’ keh dein."
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
        # A start confirmation is consent, not a symptom answer. A direct age
        # or measurement answer also starts intake and is extracted normally.
        session.screening_started = True
        if _is_start_confirmation(request.message):
            reply = _next_session_question(session)
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
    if session.screening_started and not session.awaiting_confirmation:
        unanswered = missing(session)
        if session.active_question_id not in unanswered:
            session.active_question_id = unanswered[0] if unanswered else None
    elif session.screening_started and _looks_like_question(request.message):
        reply = _answer_during_screening(request.message, session)
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request, session_id=session_id, reply=reply,
            collected=session.values, missing=missing(session), complete=False,
            awaiting_confirmation=session.awaiting_confirmation)
    correction_message = _is_correction_message(request.message)
    affirmative_confirmation = (
        session.awaiting_confirmation
        and _is_affirmative_confirmation(request.message)
        and not correction_message
    )
    negative_confirmation = (
        session.awaiting_confirmation
        and bool(re.search(r"\b(no|nope|nah|nahi|nahin|गलत|नहीं)\b", request.message.lower()))
        and not skip_targets(request.message, [])
        and not correction_message
    )
    skip_request = not session.awaiting_confirmation and is_skip_answer(request.message)
    skip_reply = None
    if skip_request:
        targets = skip_targets(request.message, session.history[:-1], session.active_question_id)
        session.skipped.update(targets)
        if any(target not in OPTIONAL for target in targets):
            skip_reply = (
                "Okay, I’ll leave that unanswered. If a required answer is skipped, I can’t safely "
                "calculate a screening estimate from incomplete inputs."
            )
    try:
        # A confirmation such as “haan, sahi hai” is not a new clinical answer.
        answer_target = session.active_question_id
        if (_is_answer_reminder(request.message) and answer_target in session.values
                and session.last_answered_question_id in session.values):
            answer_target = session.last_answered_question_id
        hair_restatement = (
            bool(re.search(r"\b(aaye|aaya|aayi)\b", request.message.lower()))
            and session.last_answered_question_id == "hair_growth"
            and session.active_question_id != "hair_growth"
            and classify_boolean_reply(request.message) is not None
        )
        if hair_restatement:
            updates = {"hair_growth": classify_boolean_reply(request.message)}
        else:
            updates = {} if affirmative_confirmation or negative_confirmation or skip_request else extract(
                request.message, session.values, session.history[:-1], answer_target
            )
    except Exception as error:
        raise HTTPException(status_code=502, detail="Feature extraction service failed: " + str(error))
    try:
        changed = merge_features(session, updates)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Invalid extracted feature values: " + str(error))

    if not session.awaiting_confirmation:
        session.intake_turns += 1

    outstanding = missing(session)
    if skip_request:
        outstanding = missing(session)

    if session.active_question_id in updates and session.active_question_id not in outstanding:
        session.last_answered_question_id = session.active_question_id
    if outstanding:
        session.active_question_id = outstanding[0]
    else:
        session.active_question_id = None

    if hair_restatement and not session.awaiting_confirmation:
        current_prompt = _question_for_active(session)
        recorded = "yes" if session.values.get("hair_growth") else "no"
        reply = f"I have your increased hair-growth answer recorded as {recorded}. Your current unanswered question is: {current_prompt}"
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request, session_id=session_id, reply=reply,
            collected=session.values, missing=outstanding, complete=False)

    if (not changed and not skip_request and not session.awaiting_confirmation
            and session.active_question_id is not None
            and session.active_question_id in outstanding):
        if session.active_question_id in BOOLEAN_FEATURES:
            clarification = "I couldn’t tell whether that was yes or no. Please answer yes/no, or say you don’t know if you’d rather skip this question."
        else:
            clarification = f"I couldn’t record a clear value for that. {_question_for_active(session)}"
        session.history.append({"role": "assistant", "content": clarification})
        return _chat_response(session, request, session_id=session_id, reply=clarification,
            collected=session.values, missing=outstanding, complete=False,
            awaiting_confirmation=False)

    if session.awaiting_confirmation:
        if changed:
            reply = _confirmation_summary(session.values, session.skipped, _uses_hinglish(session.history))
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=outstanding, complete=False, awaiting_confirmation=True,
            )
        if not _is_affirmative_confirmation(request.message):
            if re.search(r"\b(no|nope|nahi|nahin|गलत|नहीं)\b", request.message.lower()):
                reply = "Theek hai—kaunsi detail galat hai, aur uski sahi value kya hai? You can reply in Hindi or English."
            else:
                reply = (
                    "Please tell me what you’d like to correct, or confirm that the summary is accurate "
                    "so I can continue."
                )
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=outstanding, complete=False, awaiting_confirmation=True,
            )
        required_skipped = session.skipped - OPTIONAL
        if required_skipped:
            reply = (
                "I’ve left the skipped answers blank, so I can’t calculate a screening estimate without "
                + ", ".join(sorted(required_skipped)).replace("_", " ")
                + ". You can start again and provide those details if you want an estimate. This is not a diagnosis; please consult a doctor about personal concerns."
            )
            session.history.append({"role": "assistant", "content": reply})
            return _chat_response(session, request,
                session_id=session_id, reply=reply, collected=session.values,
                missing=[], complete=True, awaiting_confirmation=False,
            )
        derive_features(session)

    if outstanding:
        if session.intake_turns > MAX_INTAKE_TURNS:
            reply = fallback_question(outstanding, session.history)
        else:
            confirmed = {name: session.values[name] for name in session.values if name in session.values}
            reply = next_question(session.history, outstanding, confirmed)
        if skip_reply:
            reply = skip_reply + " " + reply
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request,
            session_id=session_id,
            reply=reply,
            collected=session.values,
            missing=outstanding,
            complete=False,
        )

    if not session.awaiting_confirmation:
        session.awaiting_confirmation = True
        reply = _confirmation_summary(session.values, session.skipped, _uses_hinglish(session.history))
        session.history.append({"role": "assistant", "content": reply})
        return _chat_response(session, request,
            session_id=session_id,
            reply=reply,
            collected=session.values,
            missing=[],
            complete=False,
            awaiting_confirmation=True,
        )

    # The prediction path remains the same; declined optional measurements are zero-filled only for model input.
    payload = {**session.values}
    for feature in OPTIONAL:
        if feature not in payload:
            payload[feature] = 0.0
    # The current measured winner remains the canonical 16-feature baseline.
    # Session-only engineered annotations must never change the model's input shape.
    from app.ml.predictor import load_assets
    _, model_features = load_assets()
    model_payload = {name: payload[name] for name in model_features}
    try:
        output = predict(model_payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(error))

    try:
        reply = explain_prediction(
            output.probability,
            output.risk_label,
            output.feature_importances,
            output.agreement_score,
            session.history,
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail="Risk explanation service failed: " + str(error))

    screening = Screening(
        user_id=user.id,
        feature_state_json={
            "features": payload,
            "feature_importances": output.feature_importances,
            "agreement_score": output.agreement_score,
            "model_probabilities": output.model_probabilities,
        },
        risk_probability=output.probability,
        risk_label=output.risk_label,
        completed=True,
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    session.screening_id = screening.id
    session.history.append({"role": "assistant", "content": reply})
    return _chat_response(session, request,
        session_id=session_id,
        reply=reply,
        collected=payload,
        missing=[],
        complete=True,
        awaiting_confirmation=False,
        prediction=output,
        screening_id=screening.id,
    )
