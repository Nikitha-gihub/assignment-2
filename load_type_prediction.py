import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_FILE = "load_data(1).csv"
RANDOM_STATE = 42

# 1. Load
df = pd.read_csv(DATA_FILE)

# 2. Parse and clean
df["Date_Time"] = pd.to_datetime(df["Date_Time"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date_Time", "Load_Type"]).copy()
df = (
    df.sort_values("Date_Time")
      .drop_duplicates(subset="Date_Time", keep="first")
      .reset_index(drop=True)
)

# 3. Feature engineering
def make_features(data):
    x = data.copy()
    dt = x["Date_Time"]

    x["year"] = dt.dt.year
    x["month"] = dt.dt.month
    x["day"] = dt.dt.day
    x["dayofweek"] = dt.dt.dayofweek
    x["hour"] = dt.dt.hour
    x["minute"] = dt.dt.minute
    x["dayofyear"] = dt.dt.dayofyear
    x["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    x["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    time_hours = dt.dt.hour + dt.dt.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * time_hours / 24)
    x["hour_cos"] = np.cos(2 * np.pi * time_hours / 24)

    return x.drop(columns=["Date_Time", "Load_Type"])

# 4. Last month = test set
last_month = df["Date_Time"].dt.to_period("M").max()
train_df = df[df["Date_Time"].dt.to_period("M") < last_month].copy()
test_df = df[df["Date_Time"].dt.to_period("M") == last_month].copy()

X_train = make_features(train_df)
y_train = train_df["Load_Type"]
X_test = make_features(test_df)
y_test = test_df["Load_Type"]

# 5. Model
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluation
print(f"Test month: {last_month}")
print(f"Training rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")
print()

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
