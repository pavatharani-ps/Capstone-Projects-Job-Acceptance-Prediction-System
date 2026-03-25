# 📊 Job Acceptance Prediction System

## 📌 Project Overview

The **Job Acceptance Prediction System** is a data-driven project that analyzes candidate profiles and predicts whether a candidate will accept a job offer.

It combines:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Models
* Interactive Dashboard using Streamlit

---

## 🎯 Objectives

* Analyze factors influencing job acceptance
* Build predictive models for placement outcomes
* Provide insights using interactive dashboards
* Help recruiters make data-driven decisions

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **XGBoost**
* **Streamlit**
* **MySQL (Optional for storage)**

---

## 📂 Project Structure

```
job_acceptance_prediction_system/
│
├── excel/
│   ├── HR_Job_Placement_Dataset.csv
│   └── cleaned_job_dataset.csv
│
├── main.py              # ML Model Training
├── stream.py            # Streamlit Dashboard
├── mysql.py             # Database Integration (Optional)
└── README.md
```

---

## 🔄 Workflow

### 1️⃣ Data Cleaning & Preprocessing

* Handled missing values (mean/median/mode)
* Encoding categorical variables
* Feature scaling (for ML models only)
* Fixed inconsistent data

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Interview Score vs Job Acceptance
* Skills Match vs Placement
* Company Tier vs Acceptance Rate
* Correlation Analysis

---

### 3️⃣ Feature Engineering

* Experience category (Fresher / Junior / Senior)
* Skills match level
* Interview performance category

---

### 4️⃣ Machine Learning Models

Models used:

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* Gradient Boosting
* **XGBoost (Best Model)** ✅

📈 **Best Accuracy Achieved: ~0.90**

---

## 📊 Streamlit Dashboard

### Key KPIs:

* Total Candidates
* Placement Rate (%)
* Job Acceptance Rate (%)
* Average Interview Score
* Average Skills Match %
* Offer Dropout Rate
* High-Risk Candidate Percentage

---

## ▶️ How to Run the Project

### 🔹 Step 1: Install Requirements

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit pymysql sqlalchemy
```

---

### 🔹 Step 2: Run ML Model

```
python main.py
```

---

### 🔹 Step 3: Run Dashboard

```
streamlit run stream.py
```

---

## 📈 Sample Output

* Model Accuracy: **~90% (XGBoost)**
* Interactive dashboard with KPI cards and analytics

---

## 🚀 Future Enhancements

* Add real-time prediction form
* Deploy on cloud (Streamlit Cloud / AWS)
* Advanced visualizations
* Model explainability (SHAP)

---
