# FallGuard — Motion Intelligence System

A full-stack fall detection web app. The Flask backend serves predictions from your trained Random Forest model. The frontend reads live mobile sensor data and classifies activity in real time.

---

## Repository Structure

```
fallguard/
├── app.py                  ← Flask backend (REST API + static file server)
├── model.pkl               ← Trained RandomForest model  (add yourself)
├── label_encoder.pkl       ← LabelEncoder               (add yourself)
├── Train.csv               ← Training dataset            (add yourself)
├── requirements.txt        ← Python dependencies
├── static/
│   └── index.html          ← Full frontend (4 tabs: Live Sensor, Manual, CSV, Stats)
└── README.md
```

> **Note:** `model.pkl`, `label_encoder.pkl`, and `Train.csv` are not committed to git (add them to `.gitignore`). Copy them into the project folder before running.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/fallguard.git
cd fallguard
pip install -r requirements.txt
```

### 2. Add your model files

Copy `model.pkl`, `label_encoder.pkl`, and `Train.csv` into the project root.

### 3. Run the server

```bash
python app.py
```

Server starts at **http://localhost:5000**

To allow phones on your local network to connect:

```bash
python app.py --host 0.0.0.0 --port 5000
```

Then open `http://YOUR_LOCAL_IP:5000` on your phone (e.g. `http://192.168.1.5:5000`).

---

## Frontend Tabs

| Tab | Description |
|-----|-------------|
| **Live Sensor** | Reads phone accelerometer + gyroscope via `DeviceMotionEvent`. Computes 9 features from a 50-sample rolling window and predicts every ~2.5 sec. |
| **Manual Input** | Slider-based input for all 9 features. Good for testing specific values. |
| **CSV Upload** | Batch prediction — upload a Test.csv and get a full results table. |
| **Dataset Stats** | Training data label distribution and feature list pulled from the backend. |

---

## API Endpoints

### `POST /api/predict`
Single prediction from JSON sensor features.

**Request:**
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
  "top5": [{"label": "FOL", "name": "Forward Fall", "prob": 87.5}, ...]
}
```

### `POST /api/predict-csv`
Batch prediction. Send a multipart form with field `file` containing your CSV.

### `GET /api/stats`
Returns training dataset statistics and label distribution.

---

## How Live Sensor Detection Works

1. Browser requests `DeviceMotionEvent` permission (required on iOS 13+)
2. Raw accelerometer (X, Y, Z) and gyroscope (alpha, beta, gamma) stream in at ~60 Hz
3. A rolling buffer of 50 samples is maintained
4. Every 25 new samples (~2.5 sec), 9 statistical features are computed:
   - **acc_max** — peak resultant acceleration magnitude
   - **gyro_max** — peak resultant gyroscope magnitude
   - **lin_max** — peak linear acceleration (gravity subtracted)
   - **acc_kurtosis / gyro_kurtosis** — distribution peakedness (high in falls)
   - **acc_skewness / gyro_skewness** — distribution asymmetry
   - **post_gyro_max / post_lin_max** — second-half window peaks (post-fall stillness)
5. Features are POST'd to `/api/predict`, result displayed instantly

---

## Activity Labels

| Code | Activity | Category |
|------|----------|----------|
| FOL | Forward Fall | fall |
| FKL | Fall Kneel | fall |
| SDL | Sideways Fall | fall |
| BSC | Back Fall | fall |
| STU | Stumble | stumble |
| WAL | Walking | normal |
| JOG | Jogging | normal |
| STD | Standing | normal |
| STN | Standing Still | normal |
| JUM | Jumping | normal |
| CSI | Chair Sit-In | normal |
| CSO | Chair Sit-Out | normal |
| SCH | Sit Chair | normal |

---

## Requirements

```
flask
flask-cors
pandas
scikit-learn
joblib
```

Install with: `pip install -r requirements.txt`

---

## .gitignore

```
model.pkl
label_encoder.pkl
Train.csv
Test.csv
output.csv
__pycache__/
*.pyc
.env
venv/
```
