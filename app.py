from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load artifacts ─────────────────────────────────────
try:
    model        = joblib.load(os.path.join(BASE_DIR, "stroke_model.pkl"))
    scaler       = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
except Exception as e:
    model = scaler = feature_cols = None
    init_error = str(e)


# ── Health check ───────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })


# ── Prediction endpoint ────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or feature_cols is None:
        return jsonify({
            "error": "Model not initialized",
        }), 500

    try:
        data = request.get_json()

        # ── Required raw inputs ─────────────────────────
        age          = float(data["age"])
        hypertension = int(data["hypertension"])
        heart_disease = int(data["disease"])
        glucose      = float(data["glucose"])
        bmi          = float(data["bmi"])

        gender       = data["gender"]
        married      = data["married"]
        work         = data["work"]
        residence    = data["residence"]
        smoking      = data["smoking"]

        # ── Manual feature vector (NO PANDAS) ───────────
        raw = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            f"gender_{gender}": 1,
            f"ever_married_{married}": 1,
            f"work_type_{work}": 1,
            f"Residence_type_{residence}": 1,
            f"smoking_status_{smoking}": 1,
        }

        # ── Convert to model input vector ───────────────
        input_vector = [raw.get(col, 0) for col in feature_cols]
        input_array = np.array(input_vector).reshape(1, -1)

        # ── Scale + Predict ─────────────────────────────
        scaled = scaler.transform(input_array)

        prediction = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][1])

        return jsonify({
            "prediction": prediction,
            "risk": "High Risk" if prediction == 1 else "Low Risk",
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)