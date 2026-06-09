# ── RUNTIME VENDOR OVERRIDE ─────────────────────────────────────────────────
# The serverless runtime pre-loads its own frozen joblib/sklearn into
# sys.modules from /var/task/_vendor/ before user code runs. sys.path
# manipulation alone cannot fix this — once a module is cached in sys.modules
# Python never searches sys.path for it again.
#
# Fix: (1) push site-packages to front of sys.path, then
#      (2) purge every joblib + sklearn key from sys.modules so the
#          next import resolves fresh from requirements.txt installs.
import sys, site

for _p in reversed(site.getsitepackages()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_purge_prefixes = ("joblib", "sklearn")
for _key in [k for k in sys.modules if k == "joblib" or k == "sklearn"
             or any(k.startswith(p + ".") for p in _purge_prefixes)]:
    del sys.modules[_key]
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

init_error_log = "No error recorded yet."

try:
    model        = joblib.load(os.path.join(BASE_DIR, "stroke_model.pkl"))
    scaler       = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
except Exception as e:
    import traceback
    model, scaler, feature_cols = None, None, None
    init_error_log = f"Exception Message: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():
    if model is None or scaler is None or feature_cols is None:
        return f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; border: 1px solid #ecc; background: #fff5f5; border-radius: 8px;">
            <h1 style="color: #c9302c; margin-top: 0;">Internal Server Error: Model Binaries Uninitialized</h1>
            <p style="color: #555; font-size: 16px;">The serverless function booted up successfully, but it failed to read your <code>.pkl</code> pipelines from the project root.</p>
            <h3 style="color: #333; margin-bottom: 5px;">Diagnostic System Log:</h3>
            <pre style="background: #222; color: #f8f8f2; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: Consolas, Monaco, monospace; font-size: 14px; line-height: 1.5;">{init_error_log}</pre>
        </div>
        """, 500

    try:
        # ── Raw inputs ───────────────────────────────────────────────────────
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

        # ── Build one-row DataFrame matching training columns ─────────────────
        row = pd.DataFrame([{
            "age":               age,
            "hypertension":      hypertension,
            "heart_disease":     disease,
            "avg_glucose_level": glucose,
            "bmi":               bmi,
            "gender":            gender,
            "ever_married":      married,
            "work_type":         work,
            "Residence_type":    residence,
            "smoking_status":    smoking,
        }])

        row = pd.get_dummies(
            row,
            columns=["gender", "ever_married", "work_type",
                     "Residence_type", "smoking_status"],
            drop_first=True,
        )
        row = row.reindex(columns=feature_cols, fill_value=0)

        # ── Scale & Predict ──────────────────────────────────────────────────
        features    = scaler.transform(row)
        prediction  = model.predict(features)[0]
        probability = round(model.predict_proba(features)[0][1] * 100, 2)

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        risk_level = "high"      if prediction == 1 else "low"

        return render_template(
            "result.html",
            prediction_text=risk_label,
            probability=probability,
            risk_level=risk_level,
            age=age, bmi=bmi, glucose=glucose, gender=gender,
            hypertension=hypertension, disease=disease,
            married=married, work=work, residence=residence, smoking=smoking,
        )

    except KeyError as ke:
        return f"Frontend Form Error: Missing expected input field: {str(ke)}", 400
    except Exception as e:
        return f"Runtime Inference Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)