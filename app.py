from flask import Flask, request, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "stroke_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
except Exception as e:
    model = scaler = feature_cols = None
    init_error = str(e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():

    if model is None:
        return f"Model loading failed: {init_error}", 500

    try:
        gender = request.form["gender"]
        age = int(request.form["age"])
        hypertension = int(request.form["hypertension"])
        disease = int(request.form["disease"])
        married = request.form["married"]
        work = request.form["work"]
        residence = request.form["residence"]
        glucose = float(request.form["glucose"])
        bmi = float(request.form["bmi"])
        smoking = request.form["smoking"]

        row = pd.DataFrame([{
            "age": age,
            "hypertension": hypertension,
            "heart_disease": disease,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "gender": gender,
            "ever_married": married,
            "work_type": work,
            "Residence_type": residence,
            "smoking_status": smoking,
        }])

        row = pd.get_dummies(
            row,
            columns=[
                "gender",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status"
            ],
            drop_first=True
        )

        row = row.reindex(columns=feature_cols, fill_value=0)

        features = scaler.transform(row)

        prediction = model.predict(features)[0]
        probability = round(
            model.predict_proba(features)[0][1] * 100,
            2
        )

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        risk_level = "high" if prediction == 1 else "low"

        return render_template(
            "result.html",
            prediction_text=risk_label,
            probability=probability,
            risk_level=risk_level,
            age=age,
            bmi=bmi,
            glucose=glucose,
            gender=gender,
            hypertension=hypertension,
            disease=disease,
            married=married,
            work=work,
            residence=residence,
            smoking=smoking
        )

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)