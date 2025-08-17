from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

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

        # Scale GPA, publications, interview score
        scaled = scaler.transform([[gpa, publications, interview_score]])[0]

        # Combine all structured features
        structured_features = [gender, ethnicity, university, specialty, experience] + list(scaled)

        # Vectorize resume
        resume_vec = vectorizer.transform([resume_text]).toarray()[0]

        # Final feature vector
        features = np.array(structured_features + list(resume_vec)).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
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
