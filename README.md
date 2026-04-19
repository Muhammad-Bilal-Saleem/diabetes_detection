# 🩺 Diabetes Classification App

A Streamlit web application for diabetes prediction and analysis using machine learning, built on the Pima Indians Diabetes Dataset.

## Features

- **Data Exploration** — Visualize feature distributions, correlation heatmaps, box plots, and pairplots
- **Model Training** — Train KNN and Logistic Regression classifiers with optional SMOTE oversampling to handle class imbalance
- **Prediction** — Enter patient health metrics and get a real-time diabetes risk assessment from both models
- **Interactive UI** — Adjustable parameters (test split, random state, SMOTE toggle) and downloadable cleaned data

## Demo

> Navigate through four pages: **Home → Data Exploration → Model Training → Prediction**

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-classification-app.git
cd diabetes-classification-app
```

### 2. Set up a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle credentials (first time only)

The app auto-downloads `diabetes.csv` from [this Kaggle dataset](https://www.kaggle.com/datasets/hossamhassan1/diabetes-classification) on first run.

**If you have a `~/.kaggle/kaggle.json`** already, it just works — no extra steps.

**If not:** Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`, or just enter your username and API key in the in-app form that appears on first launch.

### 5. Run the app

```bash
streamlit run app.py
```

## Dataset

The Pima Indians Diabetes Dataset contains the following features:

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years |
| Outcome | 0 = No diabetes, 1 = Diabetes |

## Models

| Model | Notes |
|---|---|
| K-Nearest Neighbors | k=19, tuned on this dataset |
| Logistic Regression | max_iter=1000, with feature importance via coefficients |

Both models use `StandardScaler` preprocessing and are evaluated with confusion matrices and full classification reports.

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn, imbalanced-learn (SMOTE)
- **Data**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Disclaimer

This application is for **educational purposes only** and should not replace professional medical advice.
