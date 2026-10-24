# Data Assistant Pro

> A Streamlit workbench that takes a raw CSV or database table through profiling, cleaning, AutoML modeling, time-series analysis and reporting — plus a FastAPI service exposing the same cleaning and quality checks.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) ![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)

## Overview

Analysts repeat the same steps on every new dataset: check quality, fix types and missing values, explore, try a few models, export results. Data Assistant Pro packages those steps into one app so a non-specialist can get from raw data to a validated model and report without writing code.

## Approach & tech stack

- **UI:** Streamlit app (`app.py`) with ten workspaces — Analysis Tools, Data Cleaning, Modeling, Export & Reports, Database Integration, Time Series, Model Monitoring, AI Insights, Performance Analytics, System Health
- **API:** FastAPI (`api_server.py`) — `/upload`, `/data-quality`, `/clean-data`, `/validate-data`, `/time-series`, session-based clean/export endpoints
- **Engine:** modular Python in `modules/` (loader, cleaning recommender, ML-ready cleaner, modeling, model evaluation & monitoring, time series, AI insights)
- **Stack:** pandas · scikit-learn · statsmodels · SciPy · Plotly · SQLAlchemy (PostgreSQL / MySQL) · FastAPI · Docker · GitHub Actions

## Key features

- Data-quality scoring and smart cleaning recommendations (missing values, types, outliers, duplicates)
- AutoML: model setup, side-by-side comparison, predictions, save/import trained models
- Time-series analysis and forecasting with statsmodels
- Anomaly detection, feature-engineering suggestions and auto-generated "data story"
- Model monitoring and performance diagnostics
- Connect directly to PostgreSQL / MySQL
- Exportable reports; Docker and docker-compose setup; Makefile for lint/format/type-check/security

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py              # UI on http://localhost:8501
uvicorn api_server:app --reload   # optional REST API
# or
docker-compose up --build
```

`make help` lists the developer targets (lint, format, type-check, security, docker).

---

## About the author

**Ibrahim Denis Fofanah** — Data Scientist · ML / GenAI (RAG, agents) · BI & analytics · MS Data Science, Pace University (2025)

[LinkedIn](https://www.linkedin.com/in/ibrahim-denis-fofanah/) · [Email](mailto:ibrahim.fofanah.ds@gmail.com) · [Live RAG portfolio project](https://moijey-copilot.vercel.app) · [GitHub](https://github.com/Denis060)
