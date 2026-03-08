"""Sample merged metrics with anomaly by time range and drop all-NaN columns."""

import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

START = "2014-04-10 00:00:00"
END = "2014-04-16 14:20:00"


def sample_merged_anomaly(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    start: str = START,
    end: str = END,
) -> pd.DataFrame:
    """
    Load merged metrics with anomaly, select rows in [start, end], drop columns
    that are entirely NaN, and save to output_path.
    """
    if input_path is None:
        input_path = _PROJECT_ROOT / "data" / "realAWSCloudwatch_merged_anomaly.csv"
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "merged_metrics_with_anomaly_samples.csv"

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    mask = (df["timestamp"] >= pd.to_datetime(start)) & (
        df["timestamp"] <= pd.to_datetime(end)
    )
    df = df.loc[mask].reset_index(drop=True)

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df = sample_merged_anomaly()
    print(f"Shape: {df.shape}, columns: {list(df.columns)}")
    print(df.head())
