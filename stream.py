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
    ["🔮 Prediction", "📊 Model Comparison", "📄 Classification Report", "📈 EDA"]
)

# =========================================================
# 🔮 PREDICTION
# =========================================================
if page == "🔮 Prediction":

    st.header("🔮 Employee Fit Prediction")

    skills = st.slider("Skills Match %", 0, 100, 50)
    tech = st.slider("Technical Score", 0, 100, 50)
    aptitude = st.slider("Aptitude Score", 0, 100, 50)
    exp = st.number_input("Years of Experience", 0, 10, 1)

    input_dict = {col: 0 for col in X.columns}

    input_dict["skills_match_percentage"] = skills
    input_dict["technical_score"] = tech
    input_dict["aptitude_score"] = aptitude
    input_dict["years_of_experience"] = exp

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):

        with st.spinner("Predicting... 🤖"):
            result = best_model.predict(input_df)[0]

        st.subheader(f"Best Model Used: {best_model_name}")

        if result == 1:
            st.success("✅ Employee is FIT for job")
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