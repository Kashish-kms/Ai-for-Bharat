from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_daytime",
    "irradiation_kwm2",
    "cloud_cover_pct",
    "temperature_c",
    "wind_speed_ms",
    "wind_speed_cubed",
    "capacity_mw",
    "generation_lag_1h",
    "generation_lag_24h",
    "generation_lag_168h",
    "rolling_mean_irradiation_24h",
]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(df, columns=["plant_type"], prefix="plant_type")
    plant_type_columns = [c for c in encoded.columns if c.startswith("plant_type_")]
    model_features = FEATURE_COLUMNS + plant_type_columns
    return encoded, model_features


def main() -> None:
    model_path = Path("models/lgbm_p50.pkl")
    data_path = Path("data/features.csv")
    output_path = Path("data/shap_summary.csv")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found: {data_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(
        data_path,
        parse_dates=["timestamp"],
        dtype={"cluster_id": "string"},
        low_memory=False,
    )
    df = df.sort_values(["timestamp", "plant_id"]).reset_index(drop=True)
    df, model_features = prepare_features(df)

    test_mask = (df["timestamp"] >= "2025-11-01") & (df["timestamp"] < "2026-01-01")
    test_df = df.loc[test_mask].copy()
    if test_df.empty:
        raise ValueError("No test rows found for Nov-Dec 2025 in data/features.csv.")

    X_test = test_df[model_features]

    # For LightGBM, pred_contrib returns per-feature SHAP values + 1 expected value column.
    shap_with_base = model.predict(X_test, pred_contrib=True)
    shap_values = shap_with_base[:, :-1]

    abs_shap = np.abs(shap_values)
    top3_idx = np.argsort(abs_shap, axis=1)[:, -3:][:, ::-1]

    summary = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].to_numpy(),
            "plant_id": test_df["plant_id"].to_numpy(),
        }
    )

    for rank in range(3):
        idx = top3_idx[:, rank]
        summary[f"feature_{rank + 1}"] = [model_features[i] for i in idx]
        summary[f"shap_{rank + 1}"] = shap_values[np.arange(len(test_df)), idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    global_importance = pd.DataFrame(
        {
            "feature": model_features,
            "mean_abs_shap": abs_shap.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    global_importance_path = Path("data/global_importance.csv")
    global_importance_path.parent.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(global_importance_path, index=False)

    print("Top 10 global features by mean absolute SHAP (test set):")
    for _, row in global_importance.head(10).iterrows():
        feature = row["feature"]
        value = float(row["mean_abs_shap"])
        print(f"{feature}: {value:.6f}")


if __name__ == "__main__":
    main()
