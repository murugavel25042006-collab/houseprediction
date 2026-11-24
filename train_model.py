print("🚀 train_model.py started")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

print("✅ Libraries imported")

# Create sample dataset
data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [1, 2, 2, 3, 3, 4],
    "Age": [20, 15, 10, 8, 5, 2],
    "Price": [3000000, 4000000, 5000000, 6500000, 7500000, 9000000]
}

df = pd.DataFrame(data)
print("✅ Dataset created")

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)
print("✅ Model trained")

# Ensure model folder exists
os.makedirs("model", exist_ok=True)
print("✅ model folder checked")

joblib.dump(model, "model/house_price_model.pkl")
print("🎉 Model saved successfully!")
