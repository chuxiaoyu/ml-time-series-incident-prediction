"""
CINECA dataset pipeline: sliding window + splits for CINECA experiments.

Scenario:
- Use `dataset_may_r162c05s02.csv` as the dataset source

Outputs under `data/CINECA/`:
- `windowed.csv`
- `train.csv`, `val.csv`, `test.csv`
"""

import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CINECA_DIR = _PROJECT_ROOT / "data" / "CINECA"

# --- Default paths ---
RAW_DATASETS_DIR = _CINECA_DIR / "raw_datasets"

TRAIN_VAL_INPUT_CSV = RAW_DATASETS_DIR / "r162c05s02" / "dataset_may_r162c05s02.csv"

OUT_WINDOWED = _CINECA_DIR / "windowed.csv"
OUT_TRAIN = _CINECA_DIR / "train.csv"
OUT_VAL = _CINECA_DIR / "val.csv"
OUT_TEST = _CINECA_DIR / "test.csv"

# --- Default parameters ---
H_STEPS = 6
W_STEPS = 2
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test ratio is derived as 1 - TRAIN_RATIO - VAL_RATIO


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = all except timestamp, label, jobs."""
    exclude = {"timestamp", "label", "jobs"}
    return [c for c in df.columns if c not in exclude]


# =============================================================================
# Step 1: Build sliding window
# =============================================================================
def build_sliding_window(
    input_path: Path,
    output_path: Path,
    h_steps: int = H_STEPS,
    w_steps: int = W_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window dataset from CINECA CSV.
    X: past h_steps of feature columns (flattened). Data already has avg/var per metric.
    y: 1 if any of next w_steps has label==1, else 0.
    """
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "jobs" in df.columns:
        df = df.drop(columns=["jobs"])

    feat_cols = _get_feature_columns(df)
    if "label" not in df.columns:
        raise ValueError("Input must contain 'label' column")

    df[feat_cols] = df[feat_cols].ffill()
    df = df.dropna(subset=feat_cols).reset_index(drop=True)

    n_features = len(feat_cols)
    X_list, y_list, ts_list = [], [], []

    for i in range(h_steps, len(df) - w_steps + 1):
        hist = df.iloc[i - h_steps : i][feat_cols].values
        X_list.append(hist.flatten())

        future_label = df.iloc[i : i + w_steps]["label"].values
        y_list.append(1 if (future_label == 1).any() else 0)
        ts_list.append(df.iloc[i]["timestamp"])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    feature_names = [f"t{s}_{c}" for s in range(1, h_steps + 1) for c in feat_cols]

    out_df = pd.DataFrame(X, columns=feature_names)
    out_df["timestamp"] = ts_list
    out_df["y"] = y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[Step 1] Sliding window → {output_path}  X={X.shape}  y distribution={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# =============================================================================
# Step 2: Random Train / Val / Test split from single windowed dataset
# =============================================================================
def split_train_val_test_random(
    windowed_path: Path = OUT_WINDOWED,
    train_out: Path = OUT_TRAIN,
    val_out: Path = OUT_VAL,
    test_out: Path = OUT_TEST,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random split (shuffled rows) of a single windowed dataset into
    train, validation and test sets.
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0 and train_ratio + val_ratio < 1.")

    df = pd.read_csv(windowed_path)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(
        f"[Step 2] Random train/val/test split → "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(f"  Train: {train_df['y'].value_counts().to_dict()}")
    print(f"  Val:   {val_df['y'].value_counts().to_dict()}")
    print(f"  Test:  {test_df['y'].value_counts().to_dict()}")
    return train_df, val_df, test_df


# =============================================================================
# Run full pipeline
# =============================================================================
def run_pipeline() -> None:
    """Build windowed dataset and randomly split into train/val/test from it."""
    if not TRAIN_VAL_INPUT_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {TRAIN_VAL_INPUT_CSV}")

    # Single source → windowed dataset
    build_sliding_window(input_path=TRAIN_VAL_INPUT_CSV, output_path=OUT_WINDOWED)

    # Random split into train/val/test from the same windowed dataset
    split_train_val_test_random(windowed_path=OUT_WINDOWED)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
