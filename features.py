from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("data/raw_generation.csv")
    output_path = Path("data/features.csv")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = df.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_daytime"] = df["hour_of_day"].between(6, 18).astype(int)

    grouped = df.groupby("plant_id", sort=False)
    df["generation_lag_1h"] = grouped["generation_mw"].shift(1)
    df["generation_lag_24h"] = grouped["generation_mw"].shift(24)
    df["generation_lag_168h"] = grouped["generation_mw"].shift(168)

    df["rolling_mean_irradiation_24h"] = (
        grouped["irradiation_kwm2"].rolling(window=24, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    df["wind_speed_cubed"] = df["wind_speed_ms"] ** 3

    lag_columns = ["generation_lag_1h", "generation_lag_24h", "generation_lag_168h"]
    df = df.dropna(subset=lag_columns).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
