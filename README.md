# Diabetes Prediction ML

A full-stack web application that predicts a patient's diabetes risk from clinical measurements and explains *why* using SHAP (SHapley Additive exPlanations). The system combines a tuned XGBoost classifier with a React dashboard that visualizes risk probability, feature impact, and vital-sign distributions.

## Features

- **Risk prediction** — XGBoost model trained on the Pima Indians Diabetes dataset, tuned via `RandomizedSearchCV`
- **Explainable AI** — SHAP TreeExplainer breaks down each prediction into per-feature contributions, so users see which vitals drove the result
- **Interactive dashboard** — React + Recharts UI with a radar chart comparing patient vitals against healthy/diabetic averages, and a SHAP impact bar chart
- **PDF export** — one-click report generation (`jsPDF` + `html2canvas`) of the full diagnosis dashboard
- **Dark/light mode**, responsive glassmorphism UI
- **Containerized deployment** — Docker Compose for local dev, GitHub Actions CI/CD pipeline pushing images to Amazon ECR

## Tech Stack

**Backend:** FastAPI, XGBoost, SHAP, scikit-learn, pandas, joblib
**Frontend:** React, Vite, Recharts, Axios, lucide-react
**Infra:** Docker, Docker Compose, Nginx, GitHub Actions, AWS ECR

## Architecture

```
frontend/  React SPA — collects patient vitals, calls the API, renders results
backend/   FastAPI service — loads the trained model + SHAP explainer, exposes /predict
.test/     Model training notebook and exploration scripts
```

The frontend submits patient vitals (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age) to the FastAPI `/predict` endpoint, which returns a probability score, a high/low risk flag, and SHAP values per feature for explainability.

## Model Details

- Trained on the Pima Indians Diabetes dataset (768 records, 8 features)
- Zero-value entries in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are treated as missing and imputed
- SMOTE applied to address class imbalance
- Hyperparameters tuned with `RandomizedSearchCV` (learning rate, max depth, min child weight, gamma, colsample)

## Getting Started

### Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000

### Run locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## API

`POST /predict`

```json
{
  "Pregnancies": 1,
  "Glucose": 110,
  "BloodPressure": 72,
  "SkinThickness": 25,
  "Insulin": 80,
  "BMI": 32.0,
  "DiabetesPedigreeFunction": 0.372,
  "Age": 29
}
```

Returns prediction probability, risk classification, and SHAP feature contributions.

`GET /health` — service health check.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
