"""Split sliding-window dataset into train / val / test by time (no shuffle) and save as CSV."""

import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Time-based split ratios: train, val, test
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test = 1 - train - val


def split_and_save(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sliding_window_dataset.csv (rows already in time order), split by time
    into train / val / test, and save as train.csv, val.csv, test.csv.
    No shuffling — first portion is train, then val, then test.
    """
    if input_path is None:
        input_path = _PROJECT_ROOT / "data" / "sliding_window_dataset.csv"
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data"

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    df = pd.read_csv(input_path)
    n = len(df)
    assert n > 0, "Dataset is empty"

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = split_and_save()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train: {train_df['y'].value_counts()}")
    print(f"Val: {val_df['y'].value_counts()}")
    print(f"Test: {test_df['y'].value_counts()}")
    print(f"Saved to data/train.csv, data/val.csv, data/test.csv")
