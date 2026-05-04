from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("data/forecasts.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)

    if df.empty:
        raise ValueError("data/forecasts.csv is empty.")

    df["date"] = df["timestamp"].dt.normalize()
    df["abs_err_p50"] = (df["actual"] - df["p50"]).abs()
    df["abs_err_naive"] = (df["actual"] - df["naive_baseline"]).abs()

    daily = (
        df.groupby("date", as_index=False)[["abs_err_p50", "abs_err_naive"]]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(daily) < 7:
        raise ValueError("Need at least 7 days of data to evaluate 7-day windows.")

    best_result: dict[str, float | pd.Timestamp] | None = None
    for start_idx in range(0, len(daily) - 6):
        window = daily.iloc[start_idx : start_idx + 7]
        our_mae = float(window["abs_err_p50"].mean())
        naive_mae = float(window["abs_err_naive"].mean())
        mae_diff = naive_mae - our_mae

        if best_result is None or mae_diff > float(best_result["mae_diff"]):
            best_result = {
                "start_date": window["date"].iloc[0],
                "end_date": window["date"].iloc[-1],
                "our_mae": our_mae,
                "naive_mae": naive_mae,
                "mae_diff": mae_diff,
            }

    assert best_result is not None

    start_date = pd.Timestamp(best_result["start_date"]).date()
    end_date = pd.Timestamp(best_result["end_date"]).date()
    our_mae = float(best_result["our_mae"])
    naive_mae = float(best_result["naive_mae"])
    improvement_pct = 0.0 if naive_mae == 0 else ((naive_mae - our_mae) / naive_mae) * 100.0

    print(f"Best 7-day window: {start_date} to {end_date}")
    print(f"Our average MAE (p50): {our_mae:.4f}")
    print(f"Naive average MAE: {naive_mae:.4f}")
    print(f"Improvement: {improvement_pct:.2f}%")


if __name__ == "__main__":
    main()
