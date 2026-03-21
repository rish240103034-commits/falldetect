from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__, static_folder="static")
CORS(app)

BASE  = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "model.pkl"))
le    = joblib.load(os.path.join(BASE, "label_encoder.pkl"))

FEATURE_COLS = [
    "acc_max", "gyro_max", "acc_kurtosis", "gyro_kurtosis",
    "lin_max", "acc_skewness", "gyro_skewness", "post_gyro_max", "post_lin_max"
]

LABEL_MAP = {
    "FOL": {"name": "Forward Fall",   "category": "fall"},
    "FKL": {"name": "Fall Kneel",     "category": "fall"},
    "SDL": {"name": "Sideways Fall",  "category": "fall"},
    "BSC": {"name": "Back Fall",      "category": "fall"},
    "STU": {"name": "Stumble",        "category": "stumble"},
    "WAL": {"name": "Walking",        "category": "normal"},
    "JOG": {"name": "Jogging",        "category": "normal"},
    "STD": {"name": "Standing",       "category": "normal"},
    "STN": {"name": "Standing Still", "category": "normal"},
    "JUM": {"name": "Jumping",        "category": "normal"},
    "CSI": {"name": "Chair Sit-In",   "category": "normal"},
    "CSO": {"name": "Chair Sit-Out",  "category": "normal"},
    "SCH": {"name": "Sit Chair",      "category": "normal"},
}


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json(force=True)
        features = {col: float(data.get(col, 0)) for col in FEATURE_COLS}
        df       = pd.DataFrame([features])
        pred     = model.predict(df)[0]
        proba    = model.predict_proba(df)[0]
        label    = le.inverse_transform([pred])[0]
        classes  = le.inverse_transform(list(range(len(proba))))
        top5     = sorted(zip(classes, proba), key=lambda x: -x[1])[:5]
        meta     = LABEL_MAP.get(label, {"name": label, "category": "unknown"})
        return jsonify({
            "label":      label,
            "name":       meta["name"],
            "category":   meta["category"],
            "confidence": round(float(max(proba)) * 100, 1),
            "top5": [
                {"label": l, "name": LABEL_MAP.get(l, {}).get("name", l), "prob": round(float(p) * 100, 1)}
                for l, p in top5
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        df      = pd.read_csv(request.files["file"])
        df      = df.drop(columns=["Unnamed: 0", "label", "fall"], errors="ignore")
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        preds   = model.predict(df[FEATURE_COLS])
        labels  = le.inverse_transform(preds)
        results, counts = [], {}
        for i, label in enumerate(labels):
            meta = LABEL_MAP.get(label, {"name": label, "category": "unknown"})
            results.append({"row": i + 1, "label": label, "name": meta["name"], "category": meta["category"]})
            counts[meta["category"]] = counts.get(meta["category"], 0) + 1
        return jsonify({"total": len(results), "results": results, "summary": counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        train_path = os.path.join(BASE, "Train.csv")
        if not os.path.exists(train_path):
            return jsonify({"error": "Train.csv not found"}), 404
        df           = pd.read_csv(train_path)
        label_counts = df["label"].value_counts().to_dict()
        enriched     = {
            k: {"count": v, "name": LABEL_MAP.get(k, {}).get("name", k),
                "category": LABEL_MAP.get(k, {}).get("category", "unknown")}
            for k, v in label_counts.items()
        }
        return jsonify({
            "total_samples":      len(df),
            "fall_percentage":    round(df["fall"].mean() * 100, 1),
            "label_distribution": enriched,
            "features":           FEATURE_COLS,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
