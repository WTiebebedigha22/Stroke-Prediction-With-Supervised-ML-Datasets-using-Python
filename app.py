from flask import Flask, request, render_template
import pickle
import numpy as np
# install all the dependencies, else the program will not work

#pip install flask
#pip install pickle
#pip install numpy

# Load trained model
model = pickle.load(open("strokesmodel.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def predict():
    if request.method == "POST":

        # ========== GET FORM DATA ==========
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']

        # ========== ENCODING ==========
        gender_male = 1 if gender == "Male" else 0
        gender_other = 1 if gender == "Other" else 0
        married_yes = 1 if married == "Yes" else 0
        Residence_type_Urban = 1 if residence == "Urban" else 0

        # Work type encoding
        work_type_Never_worked = 1 if work == "Never_worked" else 0
        work_type_Private = 1 if work == "Private" else 0
        work_type_Self_employed = 1 if work == "Self-employed" else 0
        work_type_children = 1 if work == "children" else 0

        # Smoking encoding
        smoking_status_formerly_smoked = 1 if smoking == "formerly smoked" else 0
        smoking_status_never_smoked = 1 if smoking == "never smoked" else 0
        smoking_status_smokes = 1 if smoking == "smokes" else 0

        # ========== FEATURE ARRAY ==========
        features = np.array([[age, hypertension, disease, glucose, bmi,
                              gender_male, gender_other, married_yes,
                              work_type_Never_worked, work_type_Private,
                              work_type_Self_employed, work_type_children,
                              Residence_type_Urban,
                              smoking_status_formerly_smoked,
                              smoking_status_never_smoked,
                              smoking_status_smokes]])

        # ========== PREDICTION ==========
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        # ========== RESULT FORMAT ==========
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
            risk_level=risk_level
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)