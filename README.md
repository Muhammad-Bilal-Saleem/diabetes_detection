# 🩺 DiabetesIQ

A Streamlit ML web app for diabetes risk classification built on the Pima Indians Diabetes Dataset. Trains and compares 6 classifiers with a dark-mode UI, real-time risk prediction, and auto dataset download via the Kaggle API.

## Features

- **🏠 Home** — Dataset overview, class balance chart, and export cleaned data
- **🔬 Data Explorer** — Feature distributions, box plots, correlation heatmap, and pairplot
- **🤖 Model Lab** — Train 6 classifiers with configurable settings, compare ROC curves, cross-validation scores, confusion matrices, and feature importance
- **🎯 Predict** — Enter patient metrics and get a consensus risk score from all models

## Models

| Model | CV Accuracy |
|---|---|
| Random Forest | ~77% |
| XGBoost | ~75% |
| Gradient Boosting | ~77% |
| Logistic Regression | ~76% |
| KNN | ~76% |
| 🏆 Voting Ensemble | ~78% |

All models use `StandardScaler` preprocessing, optional SMOTE oversampling, and IQR outlier capping. Feature engineering adds interaction terms (Glucose×BMI, Age×Glucose, etc.) for a meaningful accuracy boost.

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Muhammad-Bilal-Saleem/diabetes_detection.git
cd diabetes_detection
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

### 4. Kaggle credentials (first run only)

The app auto-downloads `diabetes.csv` on first launch.

- **Already have `~/.kaggle/kaggle.json`?** It just works.
- **Don't have one?** Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**, then enter your credentials in the in-app form that appears.

### 5. Run

```bash
streamlit run app.py
```

## Tech Stack

- **UI** — Streamlit (dark-mode, glassmorphism)
- **ML** — scikit-learn, XGBoost, imbalanced-learn
- **Data** — pandas, NumPy
- **Visualisation** — Matplotlib, Seaborn

## Dataset

[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/hossamhassan1/diabetes-classification) — 768 records, 8 features, binary outcome.

## Disclaimer

For educational purposes only. Not a substitute for professional medical advice.