import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# LOAD CSV FILE
# -----------------------------------------
df = pd.read_csv(r"D:\NAIYA\Codsoft Internship\TASK 3 IRIS Flower Classification\IRIS.csv")

print(df.head())

# -----------------------------------------
# ADD COLUMN NAMES IF MISSING
# -----------------------------------------
df.columns = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Species"
]

# -----------------------------------------
# ENCODE SPECIES
# -----------------------------------------
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# -----------------------------------------
# FEATURES AND TARGET
# -----------------------------------------
X = df.drop("Species", axis=1)
y = df["Species"]

# -----------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# MODEL
# -----------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------------------
# EVALUATION
# -----------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
