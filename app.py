from flask import Flask, request, jsonify
import numpy as np
import joblib
from scipy.stats import kurtosis, skew

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction (same as training)
def extract_features(window):
    acc = window[:, :3]
    gyro = window[:, 3:]

    features = [
        np.max(np.linalg.norm(acc, axis=1)),
        np.max(np.linalg.norm(gyro, axis=1)),
        kurtosis(acc.flatten()),
        kurtosis(gyro.flatten()),
        np.max(acc),
        skew(acc.flatten()),
        skew(gyro.flatten()),
        np.max(gyro[-10:]),
        np.max(acc[-10:])
    ]

    return features

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]  # list of sensor points

    window = np.array(data)
    features = extract_features(window)

    pred = model.predict([features])
    label = le.inverse_transform(pred)

    return jsonify({"prediction": label[0]})

if __name__ == "__main__":
    app.run(debug=True)
