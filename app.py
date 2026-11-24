print("🚀 app.py started")

from flask import Flask, render_template, request
import joblib
import numpy as np

print("✅ Libraries imported")

app = Flask(__name__)

# Load model
model = joblib.load("model/house_price_model.pkl")
print("✅ Model loaded")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    age = int(request.form["age"])

    data = np.array([[area, bedrooms, age]])
    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ₹ {round(prediction, 2)}"
    )

if __name__ == "__main__":
    print("🔥 Starting Flask server...")
    app.run(debug=True, host="127.0.0.1", port=5000)
