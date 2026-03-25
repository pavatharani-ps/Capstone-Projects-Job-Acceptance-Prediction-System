import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("excel/cleaned_job_dataset.csv")
df1 = pd.read_csv("excel/HR_Job_Placement_Dataset.csv")

# Fix status
df["status"] = df["status"].apply(lambda x: 1 if x > 0 else 0)

# KPIs
total_candidates = len(df)
placement_rate = (df["status"] == 1).mean() * 100
job_acceptance_rate = placement_rate
offer_dropout_rate = (df["status"] == 0).mean() * 100

avg_interview_score = df1["technical_score"].mean()
avg_skills_match = df1["skills_match_percentage"].mean()

high_risk = df[
    (df["skills_match_percentage"] < 50) | 
    (df["technical_score"] < 50)
]
high_risk_percentage = (len(high_risk) / len(df)) * 100


# ---------------- UI ----------------

# ---------------- Page Config ----------------
st.set_page_config(page_title="Job Dashboard", layout="wide")

st.title("📊 Job Acceptance Analytics Dashboard")

st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 5px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 15px;
}
.card h3 {
    margin: 0;
    font-size: 18px;
    color: #555;
}
.card h1 {
    margin: 10px 0 0 0;
    font-size: 28px;
    color: #000;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card">
    <h3>👥 Total Candidates</h3>
    <h1>{total_candidates}</h1>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
    <h3>📈 Placement Rate (%)</h3>
    <h1>{round(placement_rate,2)}</h1>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
    <h3>✅ Job Acceptance Rate (%)</h3>
    <h1>{round(job_acceptance_rate,2)}</h1>
</div>
""", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

col4.markdown(f"""
<div class="card">
    <h3>🎯 Avg Interview Score</h3>
    <h1>{round(avg_interview_score,2)}</h1>
</div>
""", unsafe_allow_html=True)

col5.markdown(f"""
<div class="card">
    <h3>🧠 Avg Skills Match %</h3>
    <h1>{round(avg_skills_match,2)}</h1>
</div>
""", unsafe_allow_html=True)

col6.markdown(f"""
<div class="card">
    <h3>⚠️ Offer Dropout Rate (%)</h3>
    <h1>{round(offer_dropout_rate,2)}</h1>
</div>
""", unsafe_allow_html=True)

col7, col8 = st.columns(2)

col7.markdown(f"""
<div class="card">
    <h3>🔴 High Risk Candidates (%)</h3>
    <h1>{round(high_risk_percentage,2)}</h1>
</div>
""", unsafe_allow_html=True)