import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from app.config import settings
from app.models.schemas import PredictResponse
from pathlib import Path

@lru_cache(maxsize=1)
def load_assets():
    with open(settings.features_path, encoding="utf-8") as f:
        features = json.load(f)
    model = joblib.load(settings.model_path)
    if len(features) != getattr(model, "n_features_in_", len(features)):
        raise RuntimeError("Model feature count does not match feature JSON")
    return model, features


@lru_cache(maxsize=1)
def load_ensemble():
    """Load the saved, seeded ensemble and verify its shared feature contract."""
    _, features = load_assets()
    ensemble_dir = Path(settings.model_path).resolve().parent / "ensemble"
    paths = sorted(ensemble_dir.glob("*.pkl"))
    if not paths:
        raise RuntimeError(f"No ensemble models found in {ensemble_dir}; run ml/train_ensemble.py")
    models = []
    for path in paths:
        model = joblib.load(path)
        if getattr(model, "n_features_in_", None) != len(features):
            raise RuntimeError(f"Ensemble model {path.name} does not match model feature count")
        names = list(getattr(model, "feature_names_in_", []))
        if names and names != features:
            raise RuntimeError(f"Ensemble model {path.name} feature order differs from JSON")
        models.append((path.stem, model))
    return models, features

def predict(values: dict) -> PredictResponse:
    models, names = load_ensemble()
    missing = [n for n in names if n not in values or values[n] is None]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    row = pd.DataFrame([[values[n] for n in names]], columns=names, dtype=float)
    model_probabilities = {}
    model_importances = []
    for model_name, model in models:
        probs = model.predict_proba(row)[0]
        classes = list(model.classes_)
        positive_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
        model_probabilities[model_name] = float(probs[positive_idx])
        model_importances.append(getattr(model, "feature_importances_", np.zeros(len(names))))

    probability = float(np.mean(list(model_probabilities.values())))
    risk_label = "elevated" if probability >= 0.5 else "lower"
    votes = [value >= 0.5 for value in model_probabilities.values()]
    majority_is_positive = sum(votes) > len(votes) / 2
    agreement_score = sum(vote == majority_is_positive for vote in votes) / len(votes)
    importances = np.mean(model_importances, axis=0)
    ranked = {name: float(value) for name, value in sorted(zip(names, importances), key=lambda x: -x[1])}
    return PredictResponse(
        probability=probability,
        risk_label=risk_label,
        feature_importances=ranked,
        agreement_score=agreement_score,
        model_probabilities=model_probabilities,
    )
