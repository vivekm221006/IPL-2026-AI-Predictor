# 🏏 IPL 2026 Winner Predictor AI

An advanced AI-powered cricket analytics and prediction platform that predicts IPL match outcomes and IPL 2026 championship probabilities using Machine Learning, XGBoost, SHAP Explainability, Monte Carlo Simulation, and an interactive Streamlit dashboard.

---

# 🌐 Live Demo

Add your deployed Streamlit link here:

```bash
https://your-streamlit-app.streamlit.app
```

---

# 📌 Project Overview

This project is a complete end-to-end sports analytics system built using:

- Machine Learning
- XGBoost
- LightGBM
- SHAP Explainability
- Monte Carlo Simulation
- Interactive Streamlit Dashboard

The system analyzes historical IPL data to:

- predict match winners
- estimate IPL title probabilities
- compare team strengths
- analyze venue impact
- analyze toss impact
- visualize player performance

---

# 🚀 Key Features

## ✅ Match Winner Prediction

Predict IPL match winners using trained ML models.

## ✅ IPL 2026 Championship Simulation

Monte Carlo simulation engine to estimate title probabilities.

## ✅ Player Analytics

- Top batsmen
- Top bowlers
- Strike rate analysis
- Wicket statistics

## ✅ Venue Analytics

- Venue win history
- Team dominance by stadium
- Venue-based trends

## ✅ Toss Impact Analysis

Analyze toss influence on match outcomes.

## ✅ Explainable AI (SHAP)

Understand model behavior and feature importance.

## ✅ Interactive Dashboard

Modern Streamlit-based cricket analytics dashboard.

---

# 🧠 Technologies Used

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Core Programming           |
| Pandas       | Data Processing            |
| NumPy        | Numerical Computing        |
| Scikit-learn | ML Pipeline                |
| XGBoost      | Main Prediction Model      |
| LightGBM     | Gradient Boosting          |
| SHAP         | Explainable AI             |
| Plotly       | Interactive Visualizations |
| Streamlit    | Dashboard                  |
| SQLite       | Database Support           |
| Matplotlib   | Data Visualization         |

---

# 📂 Project Structure

```bash
IPL-2026-AI-Predictor/
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
│   ├── player_analysis.py
│   ├── venue_analysis.py
│   ├── toss_analysis.py
│   ├── model_training.py
│   ├── explainability.py
│   ├── simulation.py
│   ├── live_api.py
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
├── assets/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/vivekm221006/IPL-2026-AI-Predictor.git

cd IPL-2026-AI-Predictor
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Data Loading

```bash
python src/data_loader.py
```

## Step 2 — Data Preprocessing

```bash
python src/preprocessing.py
```

## Step 3 — Feature Engineering

```bash
python src/feature_engineering.py
```

## Step 4 — Advanced Feature Engineering

```bash
python src/advanced_features.py
```

## Step 5 — Train ML Models

```bash
python src/model_training.py
```

## Step 6 — SHAP Explainability

```bash
python src/explainability.py
```

## Step 7 — Monte Carlo Simulation

```bash
python src/simulation.py
```

## Step 8 — Run Streamlit Dashboard

```bash
python -m streamlit run app.py
```

---

# 📊 Machine Learning Models

| Model         | Purpose                |
| ------------- | ---------------------- |
| Random Forest | Baseline Ensemble      |
| Extra Trees   | Advanced Tree Ensemble |
| XGBoost       | Main Prediction Model  |
| LightGBM      | Gradient Boosting      |

---

# 🧠 Explainable AI using SHAP

The project uses SHAP (SHapley Additive exPlanations) to:

- analyze feature importance
- explain predictions
- improve model transparency

Generated output:

```bash
outputs/plots/shap_summary.png
```

---

# 🎲 Monte Carlo Simulation

The project simulates complete IPL seasons multiple times to estimate championship probabilities.

Generated output:

```bash
outputs/predictions/ipl_2026_predictions.csv
```

Example:

| Team                        | Title Probability |
| --------------------------- | ----------------- |
| Chennai Super Kings         | 21.4%             |
| Mumbai Indians              | 18.2%             |
| Royal Challengers Bangalore | 15.1%             |

---

# 🖥️ Dashboard Features

## 🔮 Match Predictor

Predict IPL match winners using AI.

## 📊 IPL Winner Probability

Visualize championship probabilities.

## ⚔️ Team Comparison

Compare:

- win percentage
- strike rate
- wickets
- economy

## 👤 Player Analytics

Analyze:

- top batsmen
- top bowlers
- player performance

## 🏟️ Venue Analytics

Analyze venue-specific trends.

## 🪙 Toss Analytics

Analyze toss impact on results.

## 🧠 SHAP Analysis

Visual AI explainability dashboard.

---

# 📸 Dashboard Preview

Add screenshots inside `assets/` folder.

Example:

```markdown
![Dashboard](assets/dashboard.png)
```

---

# 📈 Future Upgrades

## 🔴 Real-Time IPL API Integration

- Live scores
- Live commentary
- Real-time analytics

---

## 🔴 Live Match Win Probability

Predict match outcome dynamically during live matches.

Example:

```text
RCB 145/3 (16 overs)

AI Win Probability:
RCB → 72%
MI → 28%
```

---

## 🔴 Fantasy Cricket Recommendation Engine

Generate:

- fantasy XI
- captain suggestions
- vice captain suggestions

---

## 🔴 Player-Level AI Predictions

- top scorer prediction
- best bowler prediction
- orange cap prediction
- purple cap prediction

---

## 🔴 Deep Learning Models

Future integration of:

- LSTM
- Neural Networks
- Transformer-based sports prediction

---

## 🔴 Advanced Visual Analytics

- venue heatmaps
- team dominance graphs
- player form curves
- batting trend analysis

---

## 🔴 User Authentication System

- login/signup
- saved predictions
- favorite teams
- prediction history

---

# 🎯 Project Highlights

✅ End-to-End ML Pipeline  
✅ Explainable AI  
✅ Monte Carlo Simulation  
✅ Interactive Dashboard  
✅ Sports Analytics Platform  
✅ Advanced Cricket Intelligence System

---

# 👨‍💻 Author

## Vivek

AI & Machine Learning Enthusiast  
CSE (AIML)

GitHub:

https://github.com/vivekm221006

---

# 📜 License

This project is for educational and research purposes.

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.
