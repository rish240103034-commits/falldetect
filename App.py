from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__, static_folder="static")
CORS(app)

# Load model and encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

FEATURE_COLS = [
    "acc_max", "gyro_max", "acc_kurtosis", "gyro_kurtosis",
    "lin_max", "acc_skewness", "gyro_skewness", "post_gyro_max", "post_lin_max"
]

LABEL_MAP = {
    "FOL": {"name": "Forward Fall", "category": "fall", "icon": "⚠️"},
    "FKL": {"name": "Fall Kneel", "category": "fall", "icon": "⚠️"},
    "SDL": {"name": "Sideways Fall", "category": "fall", "icon": "⚠️"},
    "BSC": {"name": "Back Fall", "category": "fall", "icon": "⚠️"},
    "STU": {"name": "Stumble", "category": "stumble", "icon": "⚡"},
    "WAL": {"name": "Walking", "category": "normal", "icon": "✅"},
    "JOG": {"name": "Jogging", "category": "normal", "icon": "✅"},
    "STD": {"name": "Standing", "category": "normal", "icon": "✅"},
    "STN": {"name": "Standing Still", "category": "normal", "icon": "✅"},
    "JUM": {"name": "Jumping", "category": "normal", "icon": "✅"},
    "CSI": {"name": "Chair Sit-In", "category": "normal", "icon": "✅"},
    "CSO": {"name": "Chair Sit-Out", "category": "normal", "icon": "✅"},
    "SCH": {"name": "Sit Chair", "category": "normal", "icon": "✅"},
}


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = {col: float(data.get(col, 0)) for col in FEATURE_COLS}
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        label = le.inverse_transform([pred])[0]
        classes = le.inverse_transform(list(range(len(proba))))
        top = sorted(zip(classes, proba), key=lambda x: -x[1])[:5]
        meta = LABEL_MAP.get(label, {"name": label, "category": "unknown", "icon": "❓"})
        return jsonify({
            "label": label,
            "name": meta["name"],
            "category": meta["category"],
            "icon": meta["icon"],
            "confidence": round(float(max(proba)) * 100, 1),
            "top5": [{"label": l, "name": LABEL_MAP.get(l, {}).get("name", l), "prob": round(float(p)*100,1)} for l,p in top]
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        df = pd.read_csv(f)
        df = df.drop(columns=["Unnamed: 0", "label", "fall"], errors="ignore")
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        preds = model.predict(df[FEATURE_COLS])
        labels = le.inverse_transform(preds)
        results = []
        for i, (label, row) in enumerate(zip(labels, df.itertuples())):
            meta = LABEL_MAP.get(label, {"name": label, "category": "unknown", "icon": "❓"})
            results.append({
                "row": i + 1,
                "label": label,
                "name": meta["name"],
                "category": meta["category"],
            })
        counts = {}
        for r in results:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
        return jsonify({"total": len(results), "results": results, "summary": counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        df = pd.read_csv("/home/claude/Train.csv")
        label_counts = df["label"].value_counts().to_dict()
        fall_pct = round(df["fall"].mean() * 100, 1)
        enriched = {}
        for k, v in label_counts.items():
            meta = LABEL_MAP.get(k, {"name": k, "category": "unknown"})
            enriched[k] = {"count": v, "name": meta["name"], "category": meta["category"]}
        return jsonify({
            "total_samples": len(df),
            "fall_percentage": fall_pct,
            "label_distribution": enriched,
            "features": FEATURE_COLS
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
    app.run(debug=True, port=5000)
