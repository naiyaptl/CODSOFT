# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. LOAD DATASET

file_path = r"D:\NAIYA\Codsoft Internship\TASK 2 Movie Rating Prediction with Python\IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding='latin1')

print("Dataset Loaded Successfully!")
print(df.head())

# 3. DATA CLEANING

df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]
df.dropna(inplace=True)

# 4. FEATURE ENCODING

encoder = LabelEncoder()

df['Genre'] = encoder.fit_transform(df['Genre'])
df['Director'] = encoder.fit_transform(df['Director'])
df['Actor 1'] = encoder.fit_transform(df['Actor 1'])
df['Actor 2'] = encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = encoder.fit_transform(df['Actor 3'])

# 5. FEATURE SELECTION

X = df.drop('Rating', axis=1)
y = df['Rating']

# 6. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. MODEL BUILDING

model = RandomForestRegressor(n_estimators=100, random_state=42)

# 8. MODEL TRAINING

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 9. MODEL EVALUATION

y_pred = model.predict(X_test)

print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 10. PREDICT NEW MOVIE RATING

new_movie = pd.DataFrame([[3, 120, 450, 210, 98]], columns=X.columns)
predicted_rating = model.predict(new_movie)

print("\nPredicted Movie Rating:", predicted_rating[0])