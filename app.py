from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)


model = pickle.load(open('Model/model.pkl', 'rb'))
vectorizer = pickle.load(open('Model/vectorizer.pkl', 'rb'))
scaler = pickle.load(open('Model/scaler.pkl', 'rb'))
# Load model and tools
try:
    model = pickle.load(open('Model/model.pkl', 'rb'))
    vectorizer = pickle.load(open('Model/vectorizer.pkl', 'rb'))
    scaler = pickle.load(open('Model/scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or tools: {e}")
    model = None
    vectorizer = None
    scaler = None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get structured inputs from form
        gender = int(request.form['gender'])
        ethnicity = int(request.form['ethnicity'])
        university = int(request.form['university'])
        specialty = int(request.form['specialty'])
        experience = int(request.form['experience'])

        gpa = float(request.form['gpa'])
        publications = int(request.form['publications'])
        interview_score = float(request.form['interview'])
        resume_text = request.form['resume']
        
        # Scale GPA, Publications, InterviewScore
        scaled = scaler.transform([[gpa, publications, interview_score]])[0]
        gpa_scaled, publications_scaled, interview_scaled = scaled

        # Vectorize resume (TF-IDF with max_features=15)
        resume_vec = vectorizer.transform([resume_text]).toarray()[0]

        # Build feature DataFrame in the same order as training
        # Build feature DataFrame in the same order as training
        features_df = pd.DataFrame([[  
            gender, ethnicity, university, specialty, experience,
            gpa_scaled, publications_scaled, interview_scaled, *resume_vec
        ]], columns=[
            'Gender','Ethnicity','University','Specialty','Experience',
            'GPA','Publications','InterviewScore'
        ] + list(map(str, range(len(resume_vec)))))  # TF-IDF columns


        # Predict
        prediction = model.predict(features_df)[0]
        result = "Hired ✅" if prediction == 1 else "Not Hired ❌"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', prediction_text="Error: Unable to process input.")

@app.route('/audit')
def audit():
    return render_template("bias_audit.html", image_path="static/bias_audit.png")

@app.route('/shap')
def shap_plot():
    return render_template("shap_summary.html",image_path="static/shap_summary.png")

if __name__ == '__main__':
    app.run(debug=True)
