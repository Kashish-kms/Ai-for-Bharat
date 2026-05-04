from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
TARGET_COLUMN = "generation_mw"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded = pd.get_dummies(df, columns=["plant_type"], prefix="plant_type")
    plant_type_columns = [c for c in encoded.columns if c.startswith("plant_type_")]
    model_features = FEATURE_COLUMNS + plant_type_columns
    return encoded, model_features


def train_quantile_model(
    alpha: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return model


def main() -> None:
    data_path = Path("data/features.csv")
    forecast_path = Path("data/forecasts.csv")
    model_path = Path("models/lgbm_p50.pkl")

    df = pd.read_csv(
        data_path,
        parse_dates=["timestamp"],
        dtype={"cluster_id": "string"},
        low_memory=False,
    )
    df = df.sort_values(["timestamp", "plant_id"]).reset_index(drop=True)
    df, model_features = prepare_features(df)

    train_mask = (df["timestamp"] >= "2025-01-01") & (df["timestamp"] < "2025-10-01")
    valid_mask = (df["timestamp"] >= "2025-10-01") & (df["timestamp"] < "2025-11-01")
    test_mask = (df["timestamp"] >= "2025-11-01") & (df["timestamp"] < "2026-01-01")

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        raise ValueError(
            "One or more splits are empty for requested date ranges "
            "(train Jan-Sep 2025, valid Oct 2025, test Nov-Dec 2025). "
            f"Available timestamp range is {min_ts} to {max_ts}."
        )

    X_train = train_df[model_features]
    y_train = train_df[TARGET_COLUMN]
    X_valid = valid_df[model_features]
    y_valid = valid_df[TARGET_COLUMN]
    X_test = test_df[model_features]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    model_p10 = train_quantile_model(0.1, X_train, y_train, X_valid, y_valid)
    model_p50 = train_quantile_model(0.5, X_train, y_train, X_valid, y_valid)
    model_p90 = train_quantile_model(0.9, X_train, y_train, X_valid, y_valid)

    p10 = model_p10.predict(X_test)
    p50 = model_p50.predict(X_test)
    p90 = model_p90.predict(X_test)
    naive = test_df["generation_lag_168h"].to_numpy()

    p50_rmse = rmse(y_test, p50)
    p50_mae = float(mean_absolute_error(y_test, p50))
    naive_rmse = rmse(y_test, naive)
    naive_mae = float(mean_absolute_error(y_test, naive))

    print(f"P50 RMSE: {p50_rmse:.4f}")
    print(f"P50 MAE: {p50_mae:.4f}")
    print(f"Naive RMSE: {naive_rmse:.4f}")
    print(f"Naive MAE: {naive_mae:.4f}")

    forecasts = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].to_numpy(),
            "plant_id": test_df["plant_id"].to_numpy(),
            "actual": y_test,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "naive_baseline": naive,
        }
    )
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(forecast_path, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model_p50, f)


if __name__ == "__main__":
    main()
