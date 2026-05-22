# 🏏 IPL 2026 Winner Predictor using AI & Machine Learning

An advanced AI-powered sports analytics system that predicts IPL match winners and simulates IPL 2026 tournament outcomes using Machine Learning, XGBoost, SHAP Explainability, and Monte Carlo Simulation.

---

# 📌 Project Overview

This project is a complete end-to-end IPL analytics and prediction system built using:

- Python
- Machine Learning
- XGBoost
- LightGBM
- SHAP Explainability
- Monte Carlo Simulation
- Streamlit Dashboard

The system analyzes historical IPL data and predicts:
- Match winners
- Team strengths
- IPL 2026 title probabilities

---

# 🚀 Features

## ✅ Data Pipeline
- Raw IPL dataset loading
- Data cleaning & preprocessing
- Feature engineering

## ✅ Advanced Analytics
- Team win percentage
- Recent form analysis
- Batting strength
- Bowling strength
- Economy rate
- Strike rate
- Comparative match features

## ✅ Machine Learning Models
- Random Forest
- Extra Trees
- XGBoost
- LightGBM

## ✅ Explainable AI
- SHAP feature importance analysis
- Match prediction explainability

## ✅ Monte Carlo Simulation
- Simulates complete IPL seasons
- Generates IPL 2026 championship probabilities

## ✅ Interactive Dashboard
- Match winner prediction
- IPL title probability visualization
- Team comparison dashboard
- SHAP analysis visualization

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-learn | ML Pipeline |
| XGBoost | Advanced ML Model |
| LightGBM | Gradient Boosting |
| SHAP | Explainable AI |
| Streamlit | Dashboard |
| Matplotlib | Visualization |
| SQLite | Database Support |

---

# 📂 Project Structure

```bash
ipl_2026_winner_ai/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sqlite/
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── advanced_features.py
│   ├── model_training.py
│   ├── explainability.py
│   ├── simulation.py
│   ├── db.py
│   └── utils.py
│
├── models/
│
├── outputs/
│   ├── plots/
│   ├── reports/
│   └── predictions/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore