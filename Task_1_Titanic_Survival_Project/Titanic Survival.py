# TITANIC SURVIVAL PREDICTION PROJECT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATASET
file_path = r"D:\NAIYA\Codsoft Internship\TASK 1 Titanic Survival Prediction\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print(df.head())

# 2. DATA CLEANING

# Drop columns that are not useful for prediction
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with most frequent value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 3. ENCODE CATEGORICAL DATA

label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

print("\nAfter Encoding:")
print(df.head())

# 4. FEATURE SELECTION

X = df.drop("Survived", axis=1)   # Input features
y = df["Survived"]               # Output label

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. MODEL TRAINING

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 7. MODEL EVALUATION

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. TEST WITH A NEW PASSENGER

new_passenger = np.array([[3, 1, 25, 0, 0, 7.25, 2]])
prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did NOT Survive")