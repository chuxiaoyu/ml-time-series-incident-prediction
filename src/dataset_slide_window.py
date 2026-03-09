"""Transform merged metrics with anomaly into sliding-window format for prediction."""

import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Past 30 min at 5-min resolution = 6 steps (history H)
H_STEPS = 6
# Next 10 min at 5-min resolution = 2 steps (future window W)
W_STEPS = 2


def build_sliding_windows(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    h_steps: int = H_STEPS,
    w_steps: int = W_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load merged_metrics_with_anomaly_sample(s).csv, build sliding-window samples:
    - X: past h_steps (e.g. 6) steps of value columns → h_steps * n_features (e.g. 48)
    - y: 1 if any of the next w_steps (e.g. 2) steps has anomaly, else 0

    Rows with NaN in any value column are dropped before building windows.
    Returns (X, y) and optionally saves to CSV with columns t1_value4, t1_value5, ..., t6_value17, y.
    """
    if input_path is None:
        input_path = _PROJECT_ROOT / "data" / "merged_metrics_with_anomaly_samples.csv"
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "sliding_window_dataset.csv"
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    value_cols = [c for c in df.columns if c.startswith("value")]
    if "anomaly" not in df.columns:
        raise ValueError("Input must contain an 'anomaly' column")

    # Forward-fill NaN values, then drop any remaining leading NaN rows
    df[value_cols] = df[value_cols].ffill()
    df = df.dropna(subset=value_cols).reset_index(drop=True)
    n_features = len(value_cols)
    expected_X_features = h_steps * n_features

    X_list = []
    y_list = []

    # For each position i, history is [i-h_steps, ..., i-1], future is [i, i+w_steps-1]
    # So we need i >= h_steps and i + w_steps <= len(df)
    for i in range(h_steps, len(df) - w_steps + 1):
        # History: 6 steps before current time
        hist = df.iloc[i - h_steps : i][value_cols].values  # (h_steps, n_features)
        X_list.append(hist.flatten())  # (h_steps * n_features,)
        # Next 10 min: steps i and i+1; y=1 if any anomaly in that window
        future_anomaly = df.iloc[i : i + w_steps]["anomaly"].values
        y_list.append(1 if future_anomaly.any() else 0)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    assert X.shape[1] == expected_X_features, (
        f"Expected X to have {expected_X_features} features, got {X.shape[1]}"
    )

    # Save as CSV: t1_value4, t1_value5, ..., t6_value17, y (t1 = oldest, t6 = most recent)
    feature_names = [
        f"t{step}_{col}"
        for step in range(1, h_steps + 1)
        for col in value_cols
    ]
    out_df = pd.DataFrame(X, columns=feature_names)
    out_df["y"] = y
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    return X, y


if __name__ == "__main__":
    X, y = build_sliding_windows()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(X[:2], y[:2])
