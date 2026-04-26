import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("excel/cleaned_job_dataset.csv")

# Fix column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')

# Remove duplicates
df = df.loc[:, ~df.columns.duplicated()]

# Connect MySQL
engine = create_engine("mysql+pymysql://root:db12345@localhost/job_acceptance_prediction")

# Store data
df.to_sql('job_acceptance_data', con=engine, if_exists='replace', index=False)