"""Train a seeded Random Forest ensemble without changing the canonical baseline trainer.

Run from the backend directory: python ml/train_ensemble.py
Each member uses the same canonical source columns, 80/20 seed-42 training partition,
and estimator settings as the project trainer; only random_state differs.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train_pcos_model import KEEP_RAW

ML_DIR = Path(__file__).resolve().parent
DATA_PATH = ML_DIR / "data" / "PCOS_data_without_infertility.xlsx"
FEATURES_PATH = ML_DIR / "pcos_model_features.json"
ENSEMBLE_DIR = ML_DIR / "ensemble"
SEEDS = (42, 43, 44, 45, 46)


def training_data():
    raw = pd.read_excel(DATA_PATH, sheet_name="Full_new")
    raw.columns = [str(column).strip() for column in raw.columns]
    data = raw[list(KEEP_RAW)].rename(columns=KEEP_RAW).copy()
    data["cycle_irregular"] = (data["cycle_code"] != 2).astype(int)
    data = data.drop(columns="cycle_code")
    data["bmi"] = data["weight_kg"] / (data["height_cm"] / 100) ** 2
    data["waist_hip_ratio"] = data["waist_inch"] / data["hip_inch"]
    for column in data.columns:
        if data[column].isna().any():
            known = data[column].dropna()
            fill_value = known.mode().iloc[0] if known.isin([0, 1]).all() else known.median()
            data[column] = data[column].fillna(fill_value)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return data[features], data["pcos"].astype(int), features


def main() -> None:
    X, y, features = training_data()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        destination = ENSEMBLE_DIR / f"pcos_rf_seed_{seed}.pkl"
        joblib.dump(model, destination)
        assert list(model.feature_names_in_) == features
        print(f"Saved {destination.name}: seed={seed}, rows={len(X_train)}, features={len(features)}")
    print(f"Ensemble ready: {len(SEEDS)} models in {ENSEMBLE_DIR}")


if __name__ == "__main__":
    main()
