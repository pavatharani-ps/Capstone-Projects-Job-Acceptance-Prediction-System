# 📊 Job Placement Prediction System

## 🚀 Overview

This project is a **Machine Learning-based Job Placement Prediction System** that predicts whether a candidate is **fit for a job** based on their skills, experience, and interview performance.

It also includes an **interactive Streamlit dashboard** for:

* Real-time prediction
* Model comparison
* Classification report
* Exploratory Data Analysis (EDA)

---

## 🎯 Problem Statement

Hiring the right candidate is time-consuming and sometimes inconsistent.

This system helps:

* Automate candidate screening
* Improve hiring decisions
* Reduce manual effort

---

## 🧠 Features

* 🔮 **Prediction System** (Fit / Not Fit)
* 📊 **Model Comparison** (Logistic Regression, Random Forest, XGBoost)
* 📄 **Classification Report** (Precision, Recall, F1-score)
* 📈 **EDA Dashboard** (Interactive visualizations)
* ⚡ **Fast Performance with Caching**
* ⏳ **Loading Spinners for better UX**

---

## 🗂️ Dataset

The dataset contains:

* Academic scores (SSC, HSC, Degree)
* Skills match percentage
* Technical interview score
* Aptitude score
* Years of experience
* Certifications
* Placement status (Target variable)

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Seaborn & Matplotlib

---

## 🤖 Machine Learning Models

The following models were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost (Best Performing)

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 🖥️ Streamlit Dashboard

The dashboard includes:

### 🔮 Prediction

User inputs candidate details → system predicts job suitability.

### 📊 Model Comparison

Compare accuracy of different ML models.

### 📄 Classification Report

Detailed performance metrics of the model.

### 📈 EDA

Visual insights such as:

* Skills vs placement
* Experience vs selection
* Correlation heatmap

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/job-placement-prediction.git
cd job-placement-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run stream.py
```

---

## 📁 Project Structure

```
📦 job-placement-prediction
 ┣ 📂 excel
 ┃ ┗ HR_Job_Placement_Dataset.csv
 ┣ 📜 stream.py
 ┣ 📜 main.py
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

---

## 🚀 Future Improvements

* Resume parsing integration
* Real-time data input
* Advanced UI design
* Model deployment using APIs

---

## 💡 Conclusion

This project demonstrates how Machine Learning can:

* Improve hiring efficiency
* Reduce manual workload
* Provide data-driven decisions

---

