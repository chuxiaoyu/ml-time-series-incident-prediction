"""XGBoost for CINECA incident prediction."""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # avoid segfault with sklearn/numpy

import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "CINECA"


def get_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    exclude = [c for c in ("y", "timestamp") if c in df.columns]
    X = df.drop(columns=exclude)
    y = df["y"]
    return X, y


def main() -> None:
    train = pd.read_csv(_DATA_DIR / "train.csv")
    val = pd.read_csv(_DATA_DIR / "val.csv")
    test = pd.read_csv(_DATA_DIR / "test.csv")

    X_train, y_train = get_Xy(train)
    X_val, y_val = get_Xy(val)
    X_test, y_test = get_Xy(test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        nthread=1,
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - t0
    print(f"Train time: {train_time:.3f}s\n")

    # Validation
    val_pred = model.predict(X_val)
    print("Validation results:")
    print(confusion_matrix(y_val, val_pred))
    print(classification_report(y_val, val_pred, zero_division=0))
    val_prob = model.predict_proba(X_val)[:, 1]
    print(f"Val prob — min: {val_prob.min():.4f}, max: {val_prob.max():.4f}, mean: {val_prob.mean():.4f}\n")

    # Test
    t0 = time.perf_counter()
    test_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0
    print(f"Inference time (test set, n={len(X_test)}): {inference_time:.3f}s\n")
    test_prob = model.predict_proba(X_test)[:, 1]
    print("Test results:")
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, zero_division=0))
    print(f"Test prob — min: {test_prob.min():.4f}, max: {test_prob.max():.4f}, mean: {test_prob.mean():.4f}")


if __name__ == "__main__":
    main()
