from flask import Flask, request, render_template
import os
import pickle
import pandas as pd
import numpy as np

# ─── MONKEY-PATCH FOR NUMPY CROSS-VERSION COMPATIBILITY ───
# This intercepts internal pickle operations to fix NumPy 2.x to 1.x conversion bugs
import numpy.random._pickle as nrp
from numpy.random import PCG64, MT19937, Philox, SFC64, PCG64DXSM

orig_bit_generator_ctor = nrp.__bit_generator_ctor

def patched_bit_generator_ctor(bit_generator_name='MT19937'):
    name_str = str(bit_generator_name).strip()
    
    # Detect exact or partial naming tokens inside the serialized byte stream
    if 'PCG64DXSM' in name_str:
        return PCG64DXSM()
    elif 'PCG64' in name_str:
        return PCG64()
    elif 'MT19937' in name_str:
        return MT19937()
    elif 'Philox' in name_str:
        return Philox()
    elif 'SFC64' in name_str:
        return SFC64()
    
    # If the token string is completely empty or unrecognized due to version shifts,
    # default to PCG64 (the modern baseline generator used by Scikit-Learn models)
    if not name_str:
        return PCG64()
        
    try:
        return orig_bit_generator_ctor(bit_generator_name)
    except Exception:
        return PCG64()

# Inject our custom safety net directly into the active NumPy library instance
nrp.__bit_generator_ctor = patched_bit_generator_ctor
# ──────────────────────────────────────────────────────────

app = Flask(__name__)

# Base directory for relative cloud paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variable to capture initialization tracebacks if loading fails
init_error_log = "No error recorded yet."

try:
    model = pickle.load(open(os.path.join(BASE_DIR, "stroke_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    feature_cols = pickle.load(open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb"))
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

    if request.method == "POST":
        try:
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
<<<<<<< HEAD
=======
                gender=gender,
                hypertension=hypertension,
                disease=disease,
                married=married,
                work=work,
                residence=residence,
                smoking=smoking
>>>>>>> 3d432acc50649c50b9e8e9483743140763b88fbc
            )
            
        except KeyError as ke:
            return f"Frontend Form Error: Missing expected input field form key: {str(ke)}", 400
        except Exception as e:
            return f"Runtime Inference Error: {str(e)}", 500

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
