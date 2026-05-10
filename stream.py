import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Job Dashboard", layout="wide")

st.title("📊 Job Placement ML Dashboard")

# ---------------- LOAD + TRAIN (CACHED) ----------------
@st.cache_resource
def load_and_train():

    df = pd.read_csv("excel/HR_Job_Placement_Dataset.csv")

    # Cleaning
    df.fillna(df.median(numeric_only=True), inplace=True)

    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["status"] = df["status"].map({
        "placed": 1,
        "not placed": 0,
        "yes": 1,
        "no": 0
    })

    df = pd.get_dummies(df, drop_first=True)

    # Split
    X = df.drop("status", axis=1)
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results[name] = acc
        trained_models[name] = model

    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    return df, X, X_test, y_test, results, best_model, best_model_name


# ---------------- LOAD DATA ----------------
with st.spinner("Loading data and training models... ⏳"):
    df, X, X_test, y_test, results, best_model, best_model_name = load_and_train()


# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["📊 KPI Dashboard","🔮 Prediction", "📊 Model Comparison", "📄 Classification Report", "📈 EDA"]
)

# =========================================================
# 📊 KPI DASHBOARD
# =========================================================
if page == "📊 KPI Dashboard":

    # st.header("📊 Key Performance Indicators")

    # -------- FILTERS --------
    st.sidebar.subheader("🔍 KPI Filters")

    min_exp, max_exp = st.sidebar.slider(
        "Experience Range", 0, 10, (0, 5)
    )

    min_skills = st.sidebar.slider(
        "Minimum Skills Match %", 0, 100, 50
    )

    # -------- APPLY FILTER --------
    filtered_df = df[
        (df["years_of_experience"] >= min_exp) &
        (df["years_of_experience"] <= max_exp) &
        (df["skills_match_percentage"] >= min_skills)
    ]

    # ================= KPI SECTION =================

    st.subheader("📌 Key Performance Indicators")

    # ---------------- KPI Calculations ----------------
    total = len(df)

    placed = (df["status"] == 1).sum()
    not_placed = (df["status"] == 0).sum()

    placement_rate = (placed / total) * 100

    # Job Acceptance Rate
    job_acceptance_rate = placement_rate

    # Average Scores
    avg_interview = df["technical_score"].mean()
    avg_skills = df["skills_match_percentage"].mean()

    # Offer Dropout Rate
    offer_dropout = (
        (df["status"] == 0).sum() / total
    ) * 100

    # High-Risk Candidates
    # Example: low technical + low skills
    high_risk = df[
        (df["technical_score"] < 40) &
        (df["skills_match_percentage"] < 40)
    ]

    high_risk_percent = (len(high_risk) / total) * 100


    # ================= DISPLAY KPIs =================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Candidates", total)

    col2.metric(
        "📈 Placement Rate",
        f"{placement_rate:.2f}%"
    )

    col3.metric(
        "💼 Job Acceptance Rate",
        f"{job_acceptance_rate:.2f}%"
    )

    col4.metric(
        "🎯 Avg Interview Score",
        f"{avg_interview:.2f}"
    )


    # ---------------- SECOND ROW ----------------

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "🧠 Avg Skills Match %",
        f"{avg_skills:.2f}"
    )

    col6.metric(
        "🚪 Offer Dropout Rate",
        f"{offer_dropout:.2f}%"
    )

    col7.metric(
        "⚠ High-Risk Candidates %",
        f"{high_risk_percent:.2f}%"
    )
    # -------- OPTIONAL CHART --------
    st.subheader("📊 Placement Distribution")

    chart_df = pd.DataFrame({
        "Category": ["Placed", "Not Placed"],
        "Count": [placed, not_placed]
    })

    st.bar_chart(chart_df.set_index("Category"))
# =========================================================
# 🔮 PREDICTION
# =========================================================
if page == "🔮 Prediction":

    st.header("🔮 Employee Prediction (10+ Features)")

    # -------- INPUTS --------
    age = st.slider("Age", 18, 60, 25)
    ssc = st.slider("SSC %", 0, 100, 60)
    hsc = st.slider("HSC %", 0, 100, 60)
    degree = st.slider("Degree %", 0, 100, 60)
    tech = st.slider("Technical Score", 0, 100, 50)
    aptitude = st.slider("Aptitude Score", 0, 100, 50)
    comm = st.slider("Communication Score", 0, 100, 50)
    skills = st.slider("Skills Match %", 0, 100, 50)
    exp = st.number_input("Experience (Years)", 0, 10, 1)
    cert = st.number_input("Certifications", 0, 10, 1)

    # -------- CREATE INPUT DATA --------
    input_data = {col: 0 for col in X.columns}

    # Fill important features
    input_data["age_years"] = age
    input_data["ssc_percentage"] = ssc
    input_data["hsc_percentage"] = hsc
    input_data["degree_percentage"] = degree
    input_data["technical_score"] = tech
    input_data["aptitude_score"] = aptitude
    input_data["communication_score"] = comm
    input_data["skills_match_percentage"] = skills
    input_data["years_of_experience"] = exp
    input_data["certifications_count"] = cert

    input_df = pd.DataFrame([input_data])

    # ⭐ IMPORTANT FIX (THIS SOLVES YOUR ERROR)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # -------- PREDICT --------
    if st.button("Predict"):

        result = best_model.predict(input_df)[0]

        st.subheader(f"Best Model: {best_model_name}")

        if result == 1:
            st.success("✅ Employee is FIT")
        else:
            st.error("❌ Employee is NOT FIT")


# =========================================================
# 📊 MODEL COMPARISON
# =========================================================
elif page == "📊 Model Comparison":

    st.header("📊 Model Comparison")

    with st.spinner("Preparing comparison... ⏳"):
        results_df = pd.DataFrame(
            list(results.items()), columns=["Model", "Accuracy"]
        )

    st.dataframe(results_df)
    st.bar_chart(results_df.set_index("Model"))


# =========================================================
# 📄 CLASSIFICATION REPORT
# =========================================================
elif page == "📄 Classification Report":

    st.header("📄 Classification Report")

    with st.spinner("Generating report... ⏳"):
        pred = best_model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)


# =========================================================
# 📈 EDA
# =========================================================
elif page == "📈 EDA":

    st.header("📈 Exploratory Data Analysis")

    feature = st.selectbox(
        "Select Feature",
        [
            "technical_score",
            "skills_match_percentage",
            "years_of_experience",
            "aptitude_score"
        ]
    )

    with st.spinner("Generating EDA... ⏳"):

        # BOXPLOT
        st.subheader(f"{feature} vs Placement")

        fig, ax = plt.subplots()
        sns.boxplot(x="status", y=feature, data=df, ax=ax)
        plt.title(f"{feature} vs Placement")

        st.pyplot(fig)

        # HEATMAP
        st.subheader("Correlation Heatmap")

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)

        st.pyplot(fig2)
        
