from flask import Flask, request, render_template
import pickle
import numpy as np

# ============================================================
# DEPENDENCIES
# pip install flask numpy scikit-learn
# NOTE: Do NOT pip install pickle — it is part of Python's
#       standard library and cannot/should not be installed.
# ============================================================

# ── Load trained model AND scaler ───────────────────────────
# The notebook scales features with StandardScaler before
# training. You MUST apply the same scaler at inference time,
# otherwise predictions will be completely wrong.
#
# FIX 1 – filename was "strokesmodel.pkl" in Flask but the
#          notebook saved "strokesssmodel.pkl". Make sure
#          both match. Rename the saved file or change this
#          line accordingly.
#
# FIX 2 – The scaler was NEVER saved in the notebook.
#          Add these two lines to the end of your notebook
#          and re-run it to generate scaler.pkl:
#
#              scaler_file = open('scaler.pkl', 'wb')
#              pickle.dump(scaler, scaler_file)
#              scaler_file.close()
#
model = pickle.load(open("strokesssmodel.pkl", "rb"))   # FIX 1: corrected filename
scaler = pickle.load(open("scaler.pkl", "rb"))           # FIX 2: load the scaler

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():
    if request.method == "POST":

        # ── Get form data ────────────────────────────────────
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

        # ── Encoding ─────────────────────────────────────────
        # Must match get_dummies(drop_first=True) from notebook.
        # drop_first=True removes the FIRST category alphabetically,
        # so the reference (dropped) categories are:
        #   gender       → Female
        #   ever_married → No
        #   work_type    → Govt_job
        #   Residence    → Rural
        #   smoking      → Unknown

        gender_Male = 1 if gender == "Male" else 0
        gender_Other = 1 if gender == "Other" else 0

        married_Yes = 1 if married == "Yes" else 0

        Residence_type_Urban = 1 if residence == "Urban" else 0

        work_type_Never_worked = 1 if work == "Never_worked" else 0
        work_type_Private = 1 if work == "Private" else 0
        work_type_Self_employed = 1 if work == "Self-employed" else 0
        work_type_children = 1 if work == "children" else 0

        smoking_status_formerly_smoked = 1 if smoking == "formerly smoked" else 0
        smoking_status_never_smoked = 1 if smoking == "never smoked" else 0
        smoking_status_smokes = 1 if smoking == "smokes" else 0

        # ── Build feature array ───────────────────────────────
        # Column order must exactly match x.columns from notebook:
        # age, hypertension, heart_disease, avg_glucose_level, bmi,
        # gender_Male, gender_Other, ever_married_Yes,
        # work_type_Never_worked, work_type_Private,
        # work_type_Self-employed, work_type_children,
        # Residence_type_Urban,
        # smoking_status_formerly smoked, smoking_status_never smoked,
        # smoking_status_smokes
        raw_features = np.array([[
            age, hypertension, disease, glucose, bmi,
            gender_Male, gender_Other, married_Yes,
            work_type_Never_worked, work_type_Private,
            work_type_Self_employed, work_type_children,
            Residence_type_Urban,
            smoking_status_formerly_smoked,
            smoking_status_never_smoked,
            smoking_status_smokes
        ]])

        # FIX 3 – Scale the features before prediction.
        # The model was trained on scaled data; skipping this step
        # produces nonsense predictions.
        features = scaler.transform(raw_features)

        # ── Predict ──────────────────────────────────────────
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        # ── Result ───────────────────────────────────────────
        if prediction == 1:
            risk_label = "High Risk"
            risk_level = "high"
        else:
            risk_label = "Low Risk"
            risk_level = "low"

        return render_template(
            "result.html",
            prediction_text=risk_label,
            probability=round(probability, 2),
            risk_level=risk_level,
            age=age,
            bmi=bmi,
            glucose=glucose,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)