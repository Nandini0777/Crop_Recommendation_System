# model_trainer.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load your dataset
data = pd.read_csv('crop_data.csv')  # Use your real dataset
X = data.drop('label', axis=1)
y = data['label']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)
