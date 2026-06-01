from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Base directory for relative cloud paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely load pickle assets using context-aware paths
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "stroke_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    feature_cols = pickle.load(open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb"))
except Exception as e:
    print(f"Error loading system binaries: {e}")
    model, scaler, feature_cols = None, None, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return "Internal Server Error: Model binaries are uninitialized.", 500

    if request.method == "POST":
        # ── Raw inputs ───────────────────────────────────────
        gender       = request.form["gender"]
        age          = int(request.form["age"])
        hypertension = int(request.form["hypertension"])
        disease      = int(request.form["disease"])
        married      = request.form["married"]
        work         = request.form["work"]
        residence    = request.form["residence"]
        glucose      = float(request.form["glucose"])
        bmi          = float(request.form["bmi"])
        smoking      = request.form["smoking"]

        # ── Build a one-row DataFrame matching training columns ──
        row = pd.DataFrame([{
            'age':               age,
            'hypertension':      hypertension,
            'heart_disease':     disease,
            'avg_glucose_level': glucose,
            'bmi':               bmi,
            'gender':            gender,
            'ever_married':      married,
            'work_type':         work,
            'Residence_type':    residence,
            'smoking_status':    smoking,
        }])

        row = pd.get_dummies(row, columns=['gender','ever_married','work_type',
                                            'Residence_type','smoking_status'],
                             drop_first=True)

        # Align to training feature columns (fills any missing dummies with 0)
        row = row.reindex(columns=feature_cols, fill_value=0)

        # ── Scale & Predict ──────────────────────────────────
        features   = scaler.transform(row)
        prediction = model.predict(features)[0]
        probability = round(model.predict_proba(features)[0][1] * 100, 2)

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        risk_level = "high"      if prediction == 1 else "low"

        return render_template(
            "result.html",
            prediction_text=risk_label,
            probability=probability,
            risk_level=risk_level,
            age=age,
            bmi=bmi,
            glucose=glucose,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)