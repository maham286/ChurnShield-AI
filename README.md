# 🛡️ ChurnShield AI — Enterprise Retention Intelligence & Revenue Risk Diagnostics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20LightGBM-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ChurnShield AI** is an end-to-end Machine Learning platform designed for enterprise subscription businesses to proactively identify customer churn risks, quantify financial exposure (MRR/ARR at risk), and simulate targeted retention campaign ROI in real time.

---

## 🌟 Key Features

* **Real-time Predictive Scoring:** Instant risk classification (Low, Medium, High) for uploaded customer batches using trained XGBoost / LightGBM pipelines.
* **Executive KPI Dashboard:** Real-time metrics tracking Monitored Accounts, High-Risk Customer counts, Monthly Recurring Revenue (MRR) exposure, and Annual Recurring Revenue (ARR) risk.
* **Model Explainability (SHAP):** Root-cause analysis isolating key drivers of customer attrition (e.g., contract types, tenure, monthly spend, tech support availability).
* **Prioritized At-Risk Roster:** Interactive, filterable directory allowing customer success teams to pinpoint high-risk accounts and execute targeted interventions.
* **Retention Campaign Simulator:** ROI simulator estimating saved accounts and net recovered ARR based on custom discount rates and conversion probability inputs.

---

## 🛠️ Tech Stack & Libraries

* **Frontend / Dashboard:** Streamlit, Custom Modern Dark Teal CSS, Plotly Express & Graph Objects
* **Machine Learning & Analytics:** XGBoost, LightGBM, Scikit-learn, SHAP
* **Data Engineering & Processing:** Pandas, NumPy
* **Model Persistence & Environment:** Joblib, Python 3.10+

---

## 📁 Repository Structure

```text
ChurnShield-AI/
├── data/
│   ├── raw_telecom_churn.csv        # Baseline raw customer dataset
│   ├── processed_churn_data.csv    # Cleaned customer dataset
│   └── engineered_churn_data.csv   # Feature-engineered dataset
├── notebooks/
│   ├── 01_eda_and_cleansing.ipynb        # Exploratory data analysis & data cleaning
│   ├── 02_feature_engineering.ipynb   # Encoding, scaling & domain feature creation
│   ├── 03_model_training_xgb_lgbm.ipynb # XGBoost & LightGBM model development
│   └── 04_shap_explainability.ipynb     # SHAP feature importance & model interpretability
├── saved_models/
│   ├── xgboost_churn_model.pkl    # Serialized XGBoost model pipeline
│   ├── lightgbm_churn_model.pkl   # Serialized LightGBM model pipeline
│   └── shap_summary.png           # Feature impact visualization chart
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Data loading and validation functions
│   ├── feature_engineering.py     # Feature transformation utilities
│   ├── financial_impacts.py       # MRR/ARR risk & financial loss calculations
│   └── model_utils.py             # Model inference & scoring utilities
├── .gitignore                     # Git tracking exclusions
├── app.py                         # Streamlit executive dashboard
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # Dependencies and library versions