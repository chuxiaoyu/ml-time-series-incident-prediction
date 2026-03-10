"""Random forest for CINECA incident prediction."""

import time
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "CINECA"
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
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
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    print(f"Overall — Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Test prob — min: {test_prob.min():.4f}, max: {test_prob.max():.4f}, mean: {test_prob.mean():.4f}")

    # Save summary metrics
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "train_time_sec": train_time,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        ]
    )
    summary.to_csv(_RESULTS_DIR / "rf.csv", index=False)


if __name__ == "__main__":
    main()
