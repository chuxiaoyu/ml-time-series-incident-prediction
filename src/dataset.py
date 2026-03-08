"""Transform separate metric CSV files into a merged metric CSV."""

import json
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def merge_metric_csvs(
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read all CSV files in input_dir (format: timestamp, value) and merge them
    into a single CSV with format: timestamp, value1, value2, ...

    Uses outer join on timestamp, then resamples to 5-minute intervals (mean
    within each interval). Missing values remain NaN.
    """
    if input_dir is None:
        input_dir = _PROJECT_ROOT / "data" / "realAWSCloudwatch"
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "realAWSCloudwatch_merged.csv"
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []
    for i, path in enumerate(csv_files):
        df = pd.read_csv(path)
        if "timestamp" not in df.columns or "value" not in df.columns:
            raise ValueError(
                f"Expected columns 'timestamp' and 'value' in {path.name}, got {list(df.columns)}"
            )
        df = df.rename(columns={"value": f"value{i + 1}"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs.append(df[["timestamp", f"value{i + 1}"]])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Resample to 5-minute intervals (mean within each interval)
    merged = merged.set_index("timestamp").resample("5min").mean().reset_index()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return merged


def save_metric_names_json(
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Build a mapping value1 -> metric_name, value2 -> metric_name, ... from the
    same CSV files (and order) used by merge_metric_csvs. Save as JSON.
    Metric names are the file stems (e.g. ec2_cpu_utilization_5f5533).
    """
    if input_dir is None:
        input_dir = _PROJECT_ROOT / "data" / "realAWSCloudwatch"
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "metric_names.json"
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    mapping = {f"value{i + 1}": path.stem for i, path in enumerate(csv_files)}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping


if __name__ == "__main__":
    df = merge_metric_csvs()
    print(f"Merged shape: {df.shape}")
    print(df.head(10))
    names = save_metric_names_json()
    print(f"Saved metric names to data/metric_names.json ({len(names)} entries)")
