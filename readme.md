# FallGuard — Motion Intelligence System

A full-stack fall detection web app powered by your trained Random Forest model.

## Project Structure

```
fallguard/
├── app.py              ← Flask backend (REST API)
├── model.pkl           ← Trained RandomForest model
├── label_encoder.pkl   ← Label encoder
├── Train.csv           ← Training dataset (for stats)
├── requirements.txt    ← Python dependencies
└── static/
    └── index.html      ← Frontend (single-page app)
```

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your model files

Make sure these are in the same folder as `app.py`:
- `model.pkl`
- `label_encoder.pkl`
- `Train.csv`

### 3. Run the server

```bash
python app.py
```

The app will be available at: **http://localhost:5000**

---

## API Endpoints

### `POST /api/predict`
Predict from a single set of sensor features.

**Request body (JSON):**
```json
{
  "acc_max": 26.04,
  "gyro_max": 7.31,
  "lin_max": 11.13,
  "acc_kurtosis": 20.38,
  "gyro_kurtosis": 2.78,
  "acc_skewness": 3.89,
  "gyro_skewness": 1.59,
  "post_gyro_max": 7.09,
  "post_lin_max": 10.79
}
```

**Response:**
```json
{
  "label": "FOL",
  "name": "Forward Fall",
  "category": "fall",
  "confidence": 87.5,
  "top5": [...]
}
```

---

### `POST /api/predict-csv`
Upload a CSV file and get batch predictions.

**Form field:** `file` (multipart/form-data)

---

### `GET /api/stats`
Returns training dataset statistics and label distribution.

---

## Activity Labels

| Code | Name | Category |
|------|------|----------|
| FOL | Forward Fall | fall |
| FKL | Fall Kneel | fall |
| SDL | Sideways Fall | fall |
| BSC | Back Fall | fall |
| STU | Stumble | stumble |
| WAL | Walking | normal |
| JOG | Jogging | normal |
| STD | Standing | normal |
| JUM | Jumping | normal |
| CSI | Chair Sit-In | normal |
| CSO | Chair Sit-Out | normal |
| STN | Standing Still | normal |
| SCH | Sit Chair | normal |
