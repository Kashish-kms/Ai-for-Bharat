from pathlib import Path

import pandas as pd


VALUE_COLUMNS = ["p10", "p50", "p90", "actual", "naive_baseline"]
PLANT_TO_CLUSTER = {
    "SP001": "Solar_North",
    "SP002": "Solar_North",
    "SP003": "Solar_South",
    "SP004": "Solar_South",
    "SP005": "Solar_Central",
    "WC001": "Wind_North",
    "WC002": "Wind_Central",
    "WC003": "Wind_South",
    # Backward-compatible mappings for currently generated IDs.
    "SOLAR_1": "Solar_North",
    "SOLAR_2": "Solar_North",
    "SOLAR_3": "Solar_South",
    "SOLAR_4": "Solar_South",
    "SOLAR_5": "Solar_Central",
    "WIND_1": "Wind_North",
    "WIND_2": "Wind_Central",
    "WIND_3": "Wind_South",
}


def assign_cluster_id(forecasts: pd.DataFrame) -> pd.DataFrame:
    forecasts = forecasts.copy()
    forecasts["cluster_id"] = forecasts["plant_id"].map(PLANT_TO_CLUSTER)
    if forecasts["cluster_id"].isna().any():
        unknown = sorted(forecasts.loc[forecasts["cluster_id"].isna(), "plant_id"].astype(str).unique())
        raise ValueError(f"Unmapped plant_id values found: {unknown}")
    return forecasts


def main() -> None:
    forecast_path = Path("data/forecasts.csv")
    cluster_out_path = Path("data/cluster_forecasts.csv")
    total_out_path = Path("data/total_forecasts.csv")

    forecasts = pd.read_csv(forecast_path, parse_dates=["timestamp"], low_memory=False)
    forecasts = assign_cluster_id(forecasts)

    cluster_forecasts = (
        forecasts.groupby(["timestamp", "cluster_id"], as_index=False)[VALUE_COLUMNS].sum()
        .sort_values(["timestamp", "cluster_id"])
        .reset_index(drop=True)
    )
    cluster_forecasts[VALUE_COLUMNS] = cluster_forecasts[VALUE_COLUMNS].mask(
        cluster_forecasts[VALUE_COLUMNS] < 0.001, 0.0
    )
    cluster_out_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_forecasts.to_csv(cluster_out_path, index=False)

    total_forecasts = (
        cluster_forecasts.groupby("timestamp", as_index=False)[VALUE_COLUMNS].sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    total_forecasts["cluster_id"] = "Karnataka_Total"
    total_forecasts = total_forecasts[["timestamp", "cluster_id", *VALUE_COLUMNS]]
    total_out_path.parent.mkdir(parents=True, exist_ok=True)
    total_forecasts.to_csv(total_out_path, index=False)

    print(f"cluster_forecasts shape: {cluster_forecasts.shape}")
    print(cluster_forecasts.head(5).to_string(index=False))
    print()
    print(f"total_forecasts shape: {total_forecasts.shape}")
    print(total_forecasts.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
