"""Logistic regression for CINECA incident prediction."""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Validation
    val_pred = model.predict(X_val)
    print("Validation results:")
    print(confusion_matrix(y_val, val_pred))
    print(classification_report(y_val, val_pred))
    val_prob = model.predict_proba(X_val)[:, 1]
    print(f"Val prob — min: {val_prob.min():.4f}, max: {val_prob.max():.4f}, mean: {val_prob.mean():.4f}\n")

    # Test
    test_pred = model.predict(X_test)
    print("Test results:")
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred))
    test_prob = model.predict_proba(X_test)[:, 1]
    print(f"Test prob — min: {test_prob.min():.4f}, max: {test_prob.max():.4f}, mean: {test_prob.mean():.4f}")


if __name__ == "__main__":
    main()
