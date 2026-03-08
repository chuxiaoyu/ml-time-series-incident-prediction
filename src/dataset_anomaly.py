"""Add anomaly labels to merged metrics from combined_windows.json."""

import json
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_cloudwatch_anomaly_windows(
    combined_windows_path: Path,
    metric_names: dict[str, str],
    prefix: str = "realAWSCloudwatch/",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collect all anomaly [start, end] intervals for metrics in metric_names."""
    with open(combined_windows_path) as f:
        combined_windows = json.load(f)

    intervals = []
    for stem in metric_names.values():
        key = f"{prefix}{stem}.csv"
        for window in combined_windows.get(key, []):
            start, end = pd.to_datetime(window[0]), pd.to_datetime(window[1])
            intervals.append((start, end))
    return intervals


def add_anomaly_column(
    df: pd.DataFrame,
    combined_windows_path: str | Path | None = None,
    metric_names_path: str | Path | None = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add a binary "anomaly" column (1 if timestamp falls in any labeled Cloudwatch
    anomaly window, 0 otherwise). Uses combined_windows.json and metric_names.json
    to get windows for the metrics in the dataframe.
    """
    if combined_windows_path is None:
        combined_windows_path = _PROJECT_ROOT / "data" / "combined_windows.json"
    if metric_names_path is None:
        metric_names_path = _PROJECT_ROOT / "data" / "metric_names.json"

    combined_windows_path = Path(combined_windows_path)
    if not combined_windows_path.exists():
        raise FileNotFoundError(f"Not found: {combined_windows_path}")

    with open(metric_names_path) as f:
        metric_names = json.load(f)

    intervals = _load_cloudwatch_anomaly_windows(
        combined_windows_path, metric_names
    )

    timestamps = pd.to_datetime(df[timestamp_col])
    anomaly = pd.Series(0, index=df.index, dtype=int)
    for start, end in intervals:
        anomaly |= (timestamps >= start) & (timestamps <= end)

    out = df.copy()
    out["anomaly"] = anomaly.astype(int)
    return out


def load_merged_and_add_anomaly(
    merged_csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    combined_windows_path: str | Path | None = None,
    metric_names_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the merged metrics CSV, add the "anomaly" column from combined_windows.json,
    and optionally save to a new CSV.
    """
    if merged_csv_path is None:
        merged_csv_path = _PROJECT_ROOT / "data" / "realAWSCloudwatch_merged.csv"
    merged_csv_path = Path(merged_csv_path)
    if not merged_csv_path.exists():
        raise FileNotFoundError(f"Not found: {merged_csv_path}")

    df = pd.read_csv(merged_csv_path)
    df = add_anomaly_column(
        df,
        combined_windows_path=combined_windows_path,
        metric_names_path=metric_names_path,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = load_merged_and_add_anomaly(
        output_path=_PROJECT_ROOT / "data" / "realAWSCloudwatch_merged_anomaly.csv",
    )
    print(f"Shape: {df.shape}, anomaly sum: {df['anomaly'].sum()}")
