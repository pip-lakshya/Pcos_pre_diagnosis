import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.feature_extractor import (
    extract,
    is_skip_answer,
    recover_labeled_numeric_update,
    skip_targets,
)
from app.agent.llm_client import explain_prediction, fallback_question, next_question
from app.agent.research_provider import KnowledgeBaseProvider
from app.agent.session_store import OPTIONAL, derive_features, get_session, merge_features, missing
from app.auth.security import get_current_user
from app.db.database import get_db
from app.db.models import Screening, User
from app.ml.predictor import predict
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()
MAX_INTAKE_TURNS = 15
_education_provider = KnowledgeBaseProvider()


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
    if restart_requested:
        session.screening_started = True
        session.history.append({"role": "user", "content": request.message})
        reply = next_question(session.history, missing(session), session.values)
        session.history.append({"role": "assistant", "content": reply})
        return ChatResponse(
            session_id=session_id, reply=reply, collected=session.values,
            missing=missing(session), complete=False,
        )
    if session.screening_id is not None:
        raise HTTPException(status_code=409, detail="This screening is complete; start a new session.")

    session.history.append({"role": "user", "content": request.message})
    if not session.screening_started:
        if _looks_like_question(request.message):
            reply = _answer_question_then_offer_start(request.message, session.history)
            session.history.append({"role": "assistant", "content": reply})
            return ChatResponse(
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
        if _is_declining_start(request.message):
            reply = "Theek hai, screening abhi shuru nahi karunga. Jab aap ready hon, bas ‘start’ keh dein."
            session.history.append({"role": "assistant", "content": reply})
            return ChatResponse(
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
        # A start confirmation is consent, not a symptom answer. A direct age
        # or measurement answer also starts intake and is extracted normally.
        session.screening_started = True
        if _is_start_confirmation(request.message):
            reply = next_question(session.history, missing(session), session.values)
            session.history.append({"role": "assistant", "content": reply})
            return ChatResponse(
                session_id=session_id, reply=reply, collected=session.values,
                missing=missing(session), complete=False,
            )
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
        targets = skip_targets(request.message, session.history[:-1])
        session.skipped.update(targets)
        if any(target not in OPTIONAL for target in targets):
            skip_reply = (
                "Okay, I’ll leave that unanswered. If a required answer is skipped, I can’t safely "
                "calculate a screening estimate from incomplete inputs."
            )
    try:
        # A confirmation such as “haan, sahi hai” is not a new clinical answer.
        updates = {} if affirmative_confirmation or negative_confirmation or skip_request else extract(
            request.message, session.values, session.history[:-1]
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

    if session.awaiting_confirmation:
        if changed:
            reply = _confirmation_summary(session.values, session.skipped, _uses_hinglish(session.history))
            session.history.append({"role": "assistant", "content": reply})
            return ChatResponse(
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
            return ChatResponse(
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
            return ChatResponse(
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
        return ChatResponse(
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
        return ChatResponse(
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
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        collected=payload,
        missing=[],
        complete=True,
        awaiting_confirmation=False,
        prediction=output,
        screening_id=screening.id,
    )
