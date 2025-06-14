# train_model.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load your dataset (make sure the file exists)
df = pd.read_csv("crop_data_with_season.csv")

# Features and label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']  # label should be crop name like 'rice', 'wheat', etc.

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model as pickle
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as crop_model.pkl")
