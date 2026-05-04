from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_PATH = Path("data/weather/karnataka_weather_2025.csv")
SOLAR_LOCATIONS = ["Tumkur", "Chitradurga"]
WIND_LOCATION = "Davangere"


def load_real_weather(path: Path) -> dict[str, pd.DataFrame]:
    """Load and normalize hourly weather by location."""
    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {path}. Run fetch_weather.py first to create it."
        )

    weather = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        "location_name",
        "timestamp",
        "shortwave_radiation",
        "cloudcover",
        "temperature_2m",
        "windspeed_10m",
        "winddirection_10m",
    }
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather file missing required columns: {sorted(missing)}")

    weather = weather.rename(
        columns={
            "shortwave_radiation": "irradiation_kwm2",
            "cloudcover": "cloud_cover_pct",
            "temperature_2m": "temperature_c",
            "windspeed_10m": "wind_speed_ms",
            "winddirection_10m": "wind_direction_deg",
        }
    )

    grouped: dict[str, pd.DataFrame] = {}
    for name, group in weather.groupby("location_name", sort=False):
        grouped[str(name)] = (
            group.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )

    for location in [*SOLAR_LOCATIONS, WIND_LOCATION]:
        if location not in grouped:
            raise ValueError(f"Location '{location}' not found in weather file.")

    return grouped


def solar_generation(
    capacity_mw: float,
    irradiation_kwm2: np.ndarray,
    cloud_cover_pct: np.ndarray,
    temperature_c: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Approximate utility-scale PV output from measured weather.
    shortwave_radiation from Open-Meteo is W/m^2 despite legacy naming here.
    """
    irradiance_wm2 = np.clip(irradiation_kwm2, 0.0, None)
    irradiance_factor = irradiance_wm2 / 1000.0
    performance_ratio = 0.84
    temp_factor = 1.0 - 0.0045 * np.maximum(temperature_c - 25.0, 0.0)
    cloud_factor = 1.0 - 0.15 * (cloud_cover_pct / 100.0)

    base_cf = irradiance_factor * performance_ratio * temp_factor * cloud_factor
    noise = rng.normal(0.0, 0.015, len(base_cf))
    cf = np.clip(base_cf + noise, 0.0, 1.0)
    generation = capacity_mw * cf
    generation = np.where(irradiance_wm2 <= 0.0, 0.0, generation)
    return np.clip(generation, 0.0, capacity_mw)


def wind_generation(
    capacity_mw: float, wind_speed_ms: np.ndarray, temperature_c: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Wind output from cubic power curve with temperature-based air density adjustment."""
    cut_in = 3.0
    rated = 12.0
    cut_out = 25.0

    cf = np.zeros_like(wind_speed_ms, dtype=float)
    ramp_mask = (wind_speed_ms >= cut_in) & (wind_speed_ms < rated)
    rated_mask = (wind_speed_ms >= rated) & (wind_speed_ms <= cut_out)

    cf[ramp_mask] = (wind_speed_ms[ramp_mask] ** 3 - cut_in**3) / (rated**3 - cut_in**3)
    cf[rated_mask] = 1.0

    air_density_factor = np.clip(273.15 / (temperature_c + 273.15), 0.9, 1.08)
    availability = np.clip(rng.normal(0.97, 0.01, len(wind_speed_ms)), 0.93, 1.00)
    noise = rng.normal(0.0, 0.02, len(wind_speed_ms))
    cf = np.clip(cf * air_density_factor * availability + noise, 0.0, 1.0)

    generation = capacity_mw * cf
    return np.clip(generation, 0.0, capacity_mw)


def main() -> None:
    rng = np.random.default_rng(42)
    weather_by_location = load_real_weather(WEATHER_PATH)

    records: list[pd.DataFrame] = []

    solar_capacities = rng.uniform(50, 150, 5)
    for i, cap in enumerate(solar_capacities, start=1):
        location = SOLAR_LOCATIONS[(i - 1) % len(SOLAR_LOCATIONS)]
        df = weather_by_location[location].copy()
        df["plant_id"] = f"SOLAR_{i}"
        df["plant_type"] = "solar"
        df["cluster_id"] = ""
        df["weather_location"] = location
        df["capacity_mw"] = round(float(cap), 2)
        df["generation_mw"] = solar_generation(
            capacity_mw=float(cap),
            irradiation_kwm2=df["irradiation_kwm2"].to_numpy(),
            cloud_cover_pct=df["cloud_cover_pct"].to_numpy(),
            temperature_c=df["temperature_c"].to_numpy(),
            rng=rng,
        )
        records.append(df)

    wind_capacities = rng.uniform(100, 250, 3)
    for i, cap in enumerate(wind_capacities, start=1):
        df = weather_by_location[WIND_LOCATION].copy()
        df["plant_id"] = f"WIND_{i}"
        df["plant_type"] = "wind"
        df["cluster_id"] = f"CLUSTER_{i}"
        df["weather_location"] = WIND_LOCATION
        df["capacity_mw"] = round(float(cap), 2)
        df["generation_mw"] = wind_generation(
            capacity_mw=float(cap),
            wind_speed_ms=df["wind_speed_ms"].to_numpy(),
            temperature_c=df["temperature_c"].to_numpy(),
            rng=rng,
        )
        records.append(df)

    out = pd.concat(records, ignore_index=True)
    out = out[
        [
            "timestamp",
            "plant_id",
            "plant_type",
            "cluster_id",
            "irradiation_kwm2",
            "cloud_cover_pct",
            "temperature_c",
            "wind_speed_ms",
            "wind_direction_deg",
            "weather_location",
            "capacity_mw",
            "generation_mw",
        ]
    ]

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "raw_generation.csv", index=False)

    print(f"Generated {len(out):,} rows for {out['plant_id'].nunique()} assets.")


if __name__ == "__main__":
    main()
