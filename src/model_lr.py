import pandas as pd

train = pd.read_csv("data/CINECA/train.csv")
val = pd.read_csv("data/CINECA/val.csv")
test = pd.read_csv("data/CINECA/test.csv")

def get_Xy(df):
    exclude = [c for c in ("y", "timestamp") if c in df.columns]
    X = df.drop(columns=exclude)
    y = df["y"]
    return X, y

X_train, y_train = get_Xy(train)
X_val, y_val = get_Xy(val)
X_test, y_test = get_Xy(test)

print(y_val.value_counts())
print(y_test.value_counts())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

val_pred = model.predict(X_val)

print("Validation results:")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))

test_pred = model.predict(X_test)

print("Test results:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))


test_pred = model.predict(X_test)

print("Test results:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))




# set threshold to 0.3
test_prob = model.predict_proba(X_test)[:,1]
threshold = 0.2
test_pred = (test_prob > threshold).astype(int)

print("Test results:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))

print(pd.Series(val_pred).value_counts())
print(pd.Series(test_pred).value_counts())

val_prob = model.predict_proba(X_val)[:, 1]
print(f"Val prob — min: {val_prob.min():.4f}, max: {val_prob.max():.4f}, mean: {val_prob.mean():.4f}")