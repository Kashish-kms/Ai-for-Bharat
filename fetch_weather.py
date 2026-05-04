from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
HOURLY_VARIABLES = [
    "temperature_2m",
    "cloudcover",
    "shortwave_radiation",
    "windspeed_10m",
    "winddirection_10m",
    "precipitation",
]
LOCATIONS = [
    {"location_name": "Tumkur", "lat": 13.34, "lon": 77.10, "asset_type": "solar"},
    {"location_name": "Chitradurga", "lat": 14.23, "lon": 76.40, "asset_type": "solar"},
    {"location_name": "Davangere", "lat": 14.46, "lon": 75.92, "asset_type": "wind"},
]


def fetch_hourly_weather(latitude: float, longitude: float) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    try:
        with urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Open-Meteo request failed with HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Open-Meteo API: {exc.reason}") from exc

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError("Open-Meteo response did not include 'hourly' data.")

    return hourly


def records_from_hourly(hourly: dict[str, Any]) -> list[dict[str, Any]]:
    times = hourly.get("time")
    if not isinstance(times, list):
        raise RuntimeError("Hourly data is missing 'time' values.")

    n = len(times)
    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list):
            raise RuntimeError(f"Hourly data is missing '{variable}' values.")
        if len(values) != n:
            raise RuntimeError(
                f"Mismatched series length for '{variable}': expected {n}, got {len(values)}."
            )

    rows: list[dict[str, Any]] = []
    for idx, timestamp in enumerate(times):
        row: dict[str, Any] = {"timestamp": timestamp}
        for variable in HOURLY_VARIABLES:
            row[variable] = hourly[variable][idx]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = Path("data/weather")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_location_columns = ["timestamp", *HOURLY_VARIABLES]
    merged_columns = ["location_name", "asset_type", *per_location_columns]
    merged_rows: list[dict[str, Any]] = []

    for location in LOCATIONS:
        name = str(location["location_name"])
        lat = float(location["lat"])
        lon = float(location["lon"])
        asset_type = str(location["asset_type"])

        hourly = fetch_hourly_weather(lat, lon)
        location_rows = records_from_hourly(hourly)

        safe_name = name.lower().replace(" ", "_")
        location_file = out_dir / f"{safe_name}_weather_2025.csv"
        write_csv(location_file, location_rows, per_location_columns)

        for row in location_rows:
            merged_rows.append(
                {
                    "location_name": name,
                    "asset_type": asset_type,
                    **row,
                }
            )

        print(f"Wrote {len(location_rows):,} rows -> {location_file}")

    merged_file = out_dir / "karnataka_weather_2025.csv"
    write_csv(merged_file, merged_rows, merged_columns)
    print(f"Wrote {len(merged_rows):,} merged rows -> {merged_file}")


if __name__ == "__main__":
    main()
