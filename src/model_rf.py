import pandas as pd

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


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

val_pred = model.predict(X_val)

print("\nValidation results:")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, zero_division=0))

test_pred = model.predict(X_test)

print("Test results:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, zero_division=0))


# threshold tuning
val_prob = model.predict_proba(X_val)[:, 1]
print(f"Val prob — min: {val_prob.min():.4f}, max: {val_prob.max():.4f}, mean: {val_prob.mean():.4f}")

test_prob = model.predict_proba(X_test)[:, 1]
for threshold in [0.3, 0.4, 0.5]:
    pred = (test_prob > threshold).astype(int)
    print(f"\nTest results (threshold={threshold}):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))
