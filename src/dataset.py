"""
End-to-end data processing pipeline for incident prediction.

Pipeline steps:
  1. merge_metrics        — merge per-metric CSVs into one file, resample to 5-min
  2. add_anomaly          — label each timestamp with anomaly from combined_windows.json
  3. sample               — select a time range and drop all-NaN columns
  4. build_sliding_window — create (X, y) sliding-window dataset
  5. split_dataset        — time-based train / val / test split
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# --- Default paths ---
RAW_INPUT_DIR = _DATA_DIR / "raw_NAB" / "realAWSCloudwatch"
COMBINED_WINDOWS_PATH = _DATA_DIR / "raw_NAB" / "combined_windows.json"
METRIC_NAMES_PATH = _DATA_DIR / "metric_names.json"

OUT_MERGED = _DATA_DIR / "1_merged.csv"
OUT_ANOMALY = _DATA_DIR / "2_labeled.csv"
OUT_SAMPLE = _DATA_DIR / "3_sampled.csv"
OUT_SLIDING = _DATA_DIR / "4_windowed.csv"
OUT_TRAIN = _DATA_DIR / "train.csv"
OUT_VAL = _DATA_DIR / "val.csv"
OUT_TEST = _DATA_DIR / "test.csv"

# --- Default parameters ---
SAMPLE_START = "2014-04-10 00:00:00"
SAMPLE_END = "2014-04-24 00:00:00"
H_STEPS = 6   # history: 30 min at 5-min resolution
W_STEPS = 2   # future:  10 min at 5-min resolution
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


# =============================================================================
# Step 1: Merge metrics
# =============================================================================
def merge_metrics(
    input_dir: Path = RAW_INPUT_DIR,
    output_path: Path = OUT_MERGED,
) -> pd.DataFrame:
    """Merge per-metric CSVs (timestamp, value) into one file with 5-min resampling."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    dfs = []
    for i, path in enumerate(csv_files):
        df = pd.read_csv(path)
        df = df.rename(columns={"value": f"value{i + 1}"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs.append(df[["timestamp", f"value{i + 1}"]])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.set_index("timestamp").resample("5min").mean().reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[Step 1] Merged {len(csv_files)} files → {output_path.name}  shape={merged.shape}")
    return merged


def save_metric_names_json(
    input_dir: Path = RAW_INPUT_DIR,
    output_path: Path = METRIC_NAMES_PATH,
) -> dict[str, str]:
    """Save value1→metric_name mapping as JSON (same file order as merge_metrics)."""
    csv_files = sorted(input_dir.glob("*.csv"))
    mapping = {f"value{i + 1}": p.stem for i, p in enumerate(csv_files)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


# =============================================================================
# Step 2: Add anomaly labels
# =============================================================================
def add_anomaly(
    input_path: Path = OUT_MERGED,
    output_path: Path = OUT_ANOMALY,
    combined_windows_path: Path = COMBINED_WINDOWS_PATH,
    metric_names_path: Path = METRIC_NAMES_PATH,
) -> pd.DataFrame:
    """Add binary 'anomaly' column based on labeled windows in combined_windows.json."""
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with open(metric_names_path) as f:
        metric_names = json.load(f)
    with open(combined_windows_path) as f:
        combined_windows = json.load(f)

    intervals = []
    for stem in metric_names.values():
        key = f"realAWSCloudwatch/{stem}.csv"
        for window in combined_windows.get(key, []):
            intervals.append((pd.to_datetime(window[0]), pd.to_datetime(window[1])))

    anomaly = pd.Series(0, index=df.index, dtype=int)
    for start, end in intervals:
        anomaly |= (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df["anomaly"] = anomaly.astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Step 2] Added anomaly → {output_path.name}  anomaly_sum={df['anomaly'].sum()}")
    return df


# =============================================================================
# Step 3: Sample time range
# =============================================================================
def sample(
    input_path: Path = OUT_ANOMALY,
    output_path: Path = OUT_SAMPLE,
    start: str = SAMPLE_START,
    end: str = SAMPLE_END,
) -> pd.DataFrame:
    """Select rows in [start, end] and drop all-NaN columns."""
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    mask = (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))
    df = df.loc[mask].reset_index(drop=True)
    df = df.dropna(axis=1, how="all")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Step 3] Sampled [{start} → {end}] → {output_path.name}  shape={df.shape}")
    return df


# =============================================================================
# Step 4: Sliding window
# =============================================================================
def build_sliding_window(
    input_path: Path = OUT_SAMPLE,
    output_path: Path = OUT_SLIDING,
    h_steps: int = H_STEPS,
    w_steps: int = W_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window dataset.
    X: past h_steps of value columns (flattened).  y: 1 if anomaly in next w_steps.
    """
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    value_cols = [c for c in df.columns if c.startswith("value")]
    df[value_cols] = df[value_cols].ffill()
    df = df.dropna(subset=value_cols).reset_index(drop=True)

    n_features = len(value_cols)
    X_list, y_list = [], []

    for i in range(h_steps, len(df) - w_steps + 1):
        hist = df.iloc[i - h_steps : i][value_cols].values
        X_list.append(hist.flatten())
        future_anomaly = df.iloc[i : i + w_steps]["anomaly"].values
        y_list.append(1 if future_anomaly.any() else 0)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    feature_names = [f"t{s}_{c}" for s in range(1, h_steps + 1) for c in value_cols]
    out_df = pd.DataFrame(X, columns=feature_names)
    out_df["y"] = y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[Step 4] Sliding window → {output_path.name}  X={X.shape}  y distribution={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# =============================================================================
# Step 5: Train / Val / Test split
# =============================================================================
def split_dataset(
    input_path: Path = OUT_SLIDING,
    output_dir: Path = _DATA_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split (no shuffle): train → val → test."""
    df = pd.read_csv(input_path)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"[Step 5] Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name}: {split['y'].value_counts().to_dict()}")
    return train_df, val_df, test_df


# =============================================================================
# Run full pipeline
# =============================================================================
def run_pipeline():
    """Execute all 5 steps sequentially."""
    merge_metrics()
    save_metric_names_json()
    add_anomaly()
    sample()
    build_sliding_window()
    split_dataset()
    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
