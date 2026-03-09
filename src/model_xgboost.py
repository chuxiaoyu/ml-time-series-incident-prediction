import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

X_train = train.drop(columns=["y"])
y_train = train["y"]

X_val = val.drop(columns=["y"])
y_val = val["y"]

X_test = test.drop(columns=["y"])
y_test = test["y"]

print("Train:", y_train.value_counts().to_dict())
print("Val:", y_val.value_counts().to_dict())
print("Test:", y_test.value_counts().to_dict())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


from xgboost import XGBClassifier

# scale_pos_weight compensates for class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    nthread=1,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)


from sklearn.metrics import classification_report, confusion_matrix

val_pred = model.predict(X_val)

print("\nValidation results:")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, zero_division=0))

test_pred = model.predict(X_test)

print("Test results:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, zero_division=0))


val_prob = model.predict_proba(X_val)[:, 1]
print(f"Val prob — min: {val_prob.min():.4f}, max: {val_prob.max():.4f}, mean: {val_prob.mean():.4f}")

test_prob = model.predict_proba(X_test)[:, 1]
print(f"Test prob — min: {test_prob.min():.4f}, max: {test_prob.max():.4f}, mean: {test_prob.mean():.4f}")

for threshold in [0.2, 0.3, 0.4, 0.5]:
    pred = (test_prob > threshold).astype(int)
    print(f"\nTest results (threshold={threshold}):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))
