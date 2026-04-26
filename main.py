import pandas as pd

# Load the dataset
df = pd.read_csv("./excel/HR_Job_Placement_Dataset.csv")

# Calculate missing values percentage


df["ssc_percentage"].fillna(df["ssc_percentage"].median(), inplace=True)
df["hsc_percentage"].fillna(df["hsc_percentage"].median(), inplace=True)
df["notice_period_days"].fillna(df["notice_period_days"].median(), inplace=True)
df["employment_gap_months"].fillna(df["employment_gap_months"].median(), inplace=True)

df["career_switch_willingness"].fillna(
    df["career_switch_willingness"].mode()[0], inplace=True
)
df["relevant_experience"].fillna(df["relevant_experience"].mode()[0], inplace=True)
df["job_role_match"].fillna(df["job_role_match"].mode()[0], inplace=True)

df["layoff_history"].fillna(df["layoff_history"].mode()[0], inplace=True)
df["employment_gap_months"].fillna(df["employment_gap_months"].median(), inplace=True)
df["relocation_willingness"].fillna(
    df["relocation_willingness"].mode()[0], inplace=True
)


# missing_percent = df.isnull().mean() * 100
# print(missing_percent)

# <!-----------------------------Label Encoding (for binary categories)----------------------->

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

binary_cols = [
    "gender",
    "internship_experience",
    "career_switch_willingness",
    "relevant_experience",
    "layoff_history",
    "relocation_willingness",
    "status",
]

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# <!------------------------One-Hot Encoding (for multi-category columns)------------------->

df = pd.get_dummies(
    df, columns=["degree_specialization", "company_tier"], drop_first=True
)

# <!-------------------------Feature Scaling for Numerical Columns-------------------------->

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

df[num_cols] = scaler.fit_transform(df[num_cols])


# <!---------------------------Correcting Inconsistent Categorical Labels--------------->

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].str.lower().str.strip()

# <!---------------------------Ensuring Logical Consistency Across Features--------------->

# Experience cannot be negative
df.loc[df["years_of_experience"] < 0, "years_of_experience"] = df[
    "years_of_experience"
].median()

# Notice period should not exceed 365 days
df.loc[df["notice_period_days"] > 365, "notice_period_days"] = 365

df.to_csv("excel/cleaned_job_dataset.csv", index=False)

# <!--------------------Step 4: Exploratory Data Analysis (EDA)----------------->
# <!-------------------------Interview score vs job acceptance------------>
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="status", y="technical_score", data=df)
plt.title("Interview Score vs Job Acceptance")
plt.show()

# <!------------------------Skills match percentage impact on placement------------->
sns.boxplot(x="status", y="skills_match_percentage", data=df)
plt.title("Skills Match Percentage vs Placement")
plt.show()

# <!----------------------Company tier vs acceptance rate------------------------>
# remove extra spaces
df.columns = df.columns.str.strip()

# convert one-hot to single column
df["company_tier"] = df[
    ["company_tier_Tier 1", "company_tier_Tier 2", "company_tier_Tier 3"]
].idxmax(axis=1)

# plot
sns.countplot(x="company_tier", hue="status", data=df)
plt.title("Company Tier vs Acceptance Rate")
plt.show()

# <!----------------------------Experience vs placement probability----------------->
sns.boxplot(x="status", y="years_of_experience", data=df)
plt.title("Experience vs Placement Probability")
plt.show()

# <!-----------------------------------Competition level vs job acceptance-------------->
sns.boxplot(x="status", y="competition_level", data=df)
plt.title("Competition Level vs Job Acceptance")
plt.show()

# <!-----------------------Correlation analysis among numeric features----------------->
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# <!------------------------------------Step 5: Feature Engineering--------------------->
# <!-------------------------Experience category (Fresher / Junior / Senior)---------------------------->


def experience_category(x):
    if x == 0:
        return "Fresher"
    elif x <= 3:
        return "Junior"
    else:
        return "Senior"


df["experience_category"] = df["years_of_experience"].apply(experience_category)


# <!-------------------------------------------Academic Performance Bands---------------------------->
def academic_band(x):
    if x < 60:
        return "Low"
    elif x <= 75:
        return "Medium"
    else:
        return "High"


df["academic_performance"] = df["degree_percentage"].apply(academic_band)


# <!----------------------------------Skills match level (Low / Medium / High)----------------------->
def skill_level(x):
    if x < 50:
        return "Low"
    elif x <= 75:
        return "Medium"
    else:
        return "High"


df["skills_level"] = df["skills_match_percentage"].apply(skill_level)


# <!--------------------------------------Interview performance category------------------------->
def interview_category(x):
    if x < 50:
        return "Poor"
    elif x <= 75:
        return "Average"
    else:
        return "Good"


df["interview_performance"] = df["technical_score"].apply(interview_category)

# <!----------------------------------------Placement probability score------------------------->
# remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# create placement score
df["placement_score"] = (
    0.4 * df["technical_score"]
    + 0.3 * df["skills_match_percentage"]
    + 0.3 * df["aptitude_score"]
)


# create placement category
def placement_category(x):
    if x < 50:
        return "Low"
    elif x <= 75:
        return "Medium"
    else:
        return "High"


df["placement_probability"] = df["placement_score"].apply(placement_category)

# <!---------------------------Machine Learning Modeling----------------------->

# <!-------------------------Convert Target Variable----------------------->
# ---------------- Target Conversion ----------------
y = df["status"].apply(lambda x: 1 if x > 0 else 0)

# y=df['status']

# ---------------- Features ----------------
X = df.drop("status", axis=1)

# Convert categorical → numeric
X = pd.get_dummies(X, drop_first=True)

# ---------------- Train Test Split ----------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X_scaled = scaler.fit_transform(X)

# ---------------- Train Test Split ----------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Models ----------------
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))




# <!---------------------Analyst Tasks (EDA & ML Analytics)--------------->

# Academic scores vs placement outcome
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="status", y="degree_percentage", data=df)
plt.title("Academic Scores vs Placement Outcome")
plt.show()

# Skills match vs interview performance
sns.scatterplot(x="skills_match_percentage", y="technical_score", hue="status", data=df)
plt.title("Skills Match vs Interview Performance")
plt.show()

# Certification impact on job acceptance
sns.barplot(x="certifications_count", y="status", data=df)
plt.title("Certification Impact on Job Acceptance")
plt.show()

# Acceptance rate by company tier
sns.countplot(x="company_tier", hue="status", data=df)
plt.title("Acceptance Rate by Company Tier")
plt.show()

# Experience vs placement success
sns.boxplot(x="status", y="years_of_experience", data=df)
plt.title("Experience vs Placement Success")
plt.show()

# Interview score vs placement probability
sns.boxplot(x="status", y="technical_score", data=df)
plt.title("Interview Score vs Placement Probability")
plt.show()

# Employability test score analysis
numeric_df = df.select_dtypes(include="number")

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Employability Score Correlation")
plt.show()
