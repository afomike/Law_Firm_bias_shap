# LexAI — Fair Hiring Intelligence for Law Firms

> A machine learning web application for law firm candidate outcome prediction, built with fairness and interpretability at its core.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://law-firm-bias-shap.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Model Artifacts](#model-artifacts)
- [Fairness & Explainability](#fairness--explainability)
- [Notes & Limitations](#notes--limitations)
- [License](#license)

---

## Overview

**LexAI** is a Flask-based prototype that predicts whether a law firm candidate is likely to be hired, while simultaneously surfacing bias and interpretability insights about those predictions. It is designed to support responsible AI evaluation in legal hiring workflows — not to replace human judgment, but to make algorithmic influence visible and auditable.

The application combines structured candidate data with free-form resume text, processes both through a pre-trained ML pipeline, and renders predictions alongside fairness metrics and SHAP-based explanations.

---

## Live Demo

The application is deployed and publicly accessible at:

**[https://law-firm-bias-shap.onrender.com/](https://law-firm-bias-shap.onrender.com/)**

> **Note:** The demo runs on Render's free tier and may take 30–60 seconds to wake from a cold start.

---

## Features

| Feature | Description |
|---|---|
| **Candidate Prediction** | Structured form + resume text input fed through a trained ML model |
| **Bias Audit View** | Visual analysis of model outcomes across demographic groups |
| **SHAP Explanation View** | Per-feature importance scores for model interpretability |
| **TF-IDF Resume Vectorization** | Converts free-form resume text into model-ready features |
| **Pre-trained Pipeline** | Loads serialized model, vectorizer, and scaler artifacts at startup |
| **Graceful Degradation** | Application continues running and reports errors if any artifact fails to load |

---

## Architecture

```
User Browser
     │
     ▼
Flask Application (app.py)
     │
     ├── /            →  Candidate prediction form & result
     ├── /audit       →  Bias audit visualizations
     └── /shap        →  SHAP feature explanation page
           │
           ▼
     ML Pipeline
     ├── model.pkl       (trained classifier)
     ├── vectorizer.pkl  (TF-IDF resume vectorizer)
     └── scaler.pkl      (feature scaler)
```

---

## Project Structure

```
LexAI/
│
├── app.py                  # Flask application entry point and route definitions
├── requirements.txt        # Python package dependencies
├── runtime.txt             # Target Python runtime specification
│
├── Model/
│   ├── model.pkl           # Serialized trained ML classifier
│   ├── vectorizer.pkl      # Serialized TF-IDF vectorizer for resume text
│   └── scaler.pkl          # Serialized feature scaler
│
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Candidate prediction page
│   ├── audit.html          # Bias audit page
│   └── shap.html           # SHAP explanation page
│
├── static/                 # Static assets (images, charts)
│
├── Dataset/                # Source data used during model development
│
├── README.md
└── LICENSE
```

---

## Prerequisites

- **Python 3.12** (see `runtime.txt`)
- `pip` or a compatible package manager
- All serialized model artifacts present in `Model/` (see [Model Artifacts](#model-artifacts))

---

## Installation

**1. Clone the repository.**

```bash
git clone https://github.com/your-username/lexai.git
cd lexai
```

**2. Create and activate a virtual environment.**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies.**

```bash
pip install -r requirements.txt
```

**4. Confirm model artifacts are in place.**

```bash
ls Model/
# Expected: model.pkl  vectorizer.pkl  scaler.pkl
```

---

## Running Locally

Start the development server:

```bash
python app.py
```

Then open a browser and navigate to any of the available routes:

| Route | Description |
|---|---|
| `http://127.0.0.1:5000/` | Candidate prediction form |
| `http://127.0.0.1:5000/audit` | Bias audit dashboard |
| `http://127.0.0.1:5000/shap` | SHAP model explanation view |

---

## Deployment

For production environments, serve the application using a WSGI server such as **Gunicorn**:

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

### Deploying to Render

This project is configured for deployment on [Render](https://render.com/):

1. Connect your GitHub repository to a new Render **Web Service**.
2. Set the **Build Command** to `pip install -r requirements.txt`.
3. Set the **Start Command** to `gunicorn app:app`.
4. Ensure the `Model/` directory and its `.pkl` artifacts are committed to the repository or made available via environment-specified paths.

---

## Model Artifacts

The application depends on three serialized artifacts located in `Model/`:

| File | Purpose |
|---|---|
| `model.pkl` | The trained classification model |
| `vectorizer.pkl` | TF-IDF vectorizer fitted on resume text |
| `scaler.pkl` | Feature scaler fitted on structured candidate inputs |

If any artifact fails to load at startup, the application will log an error to the console and disable the prediction functionality — the audit and SHAP pages will remain accessible.

---

## Fairness & Explainability

LexAI is built around two core transparency principles:

- **Bias Audit (`/audit`):** Visualizes model outcome distributions across candidate demographic groups to surface disparate impact and flag potential discriminatory patterns.
- **SHAP Explanations (`/shap`):** Uses SHAP (SHapley Additive exPlanations) values to decompose individual predictions into per-feature contributions, making the model's reasoning legible to non-technical reviewers.

> These tools are intended for audit and oversight purposes. LexAI is a research and demonstration prototype — it is **not** certified for use in live HR or legal hiring decisions.

---

## Notes & Limitations

- The UI is designed for demonstration, exploratory analysis, and fairness auditing — not production HR automation.
- The model reflects patterns in its training dataset and may encode historical biases. The audit view exists precisely to surface these.
- All `.pkl` artifacts must be generated from compatible versions of `scikit-learn` and related libraries matching `requirements.txt`.

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for the full terms.
