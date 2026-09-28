"""Train the canonical class-weighted Random Forest from the bundled workbook.

Run from any directory with the backend environment active:
    python ml/train_pcos_model.py
This intentionally replaces ml/pcos_rf_model.pkl and its feature-order JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ML_DIR = Path(__file__).resolve().parent
DATA_PATH = ML_DIR / "data" / "PCOS_data_without_infertility.xlsx"
MODEL_PATH = ML_DIR / "pcos_rf_model.pkl"
FEATURES_PATH = ML_DIR / "pcos_model_features.json"
KEEP_RAW = {
    "PCOS (Y/N)": "pcos",
    "Age (yrs)": "age",
    "Weight (Kg)": "weight_kg",
    "Height(Cm)": "height_cm",
    "Cycle(R/I)": "cycle_code",
    "Cycle length(days)": "cycle_length_days",
    "Weight gain(Y/N)": "weight_gain",
    "hair growth(Y/N)": "hair_growth",
    "Skin darkening (Y/N)": "skin_darkening",
    "Hair loss(Y/N)": "hair_loss",
    "Pimples(Y/N)": "acne",
    "Fast food (Y/N)": "fast_food",
    "Reg.Exercise(Y/N)": "regular_exercise",
    "Hip(inch)": "hip_inch",
    "Waist(inch)": "waist_inch",
}


def main() -> None:
    raw = pd.read_excel(DATA_PATH, sheet_name="Full_new")
    raw.columns = [c.strip() for c in raw.columns]
    data = raw[list(KEEP_RAW)].rename(columns=KEEP_RAW).copy()
    data["cycle_irregular"] = (data["cycle_code"] != 2).astype(int)
    data = data.drop(columns="cycle_code")
    data["bmi"] = data["weight_kg"] / (data["height_cm"] / 100) ** 2
    data["waist_hip_ratio"] = data["waist_inch"] / data["hip_inch"]
    for col in data.columns:
        if data[col].isna().any():
            known = data[col].dropna()
            fill = known.mode()[0] if known.isin([0, 1]).all() else known.median()
            data[col] = data[col].fillna(fill)
    data["pcos"] = data["pcos"].astype(int)
    feature_cols = [c for c in data.columns if c != "pcos"]
    X_train, _, y_train, _ = train_test_split(
        data[feature_cols], data["pcos"], test_size=0.2, stratify=data["pcos"], random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2) + "\n", encoding="utf-8")
    print(f"Saved canonical model: {MODEL_PATH}")
    print(f"Saved feature order: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
