import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print(df.isnull().sum())
# Check blank values in TotalCharges
print("Blank values in TotalCharges:",
      (df["TotalCharges"] == " ").sum())

print(df.dtypes)
# convert TotalCharges columns in numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
print(df.isnull().sum())
# drop missing value
df.dropna(inplace=True)
print(df.isnull().sum())
# chack duplicate
print("duplicated",df.duplicated().sum())
# drop Customer id columns
df.drop("customerID",axis=1,inplace=True)
print(df.columns)
print(df.dtypes)

print("Dataset Cleaned 👍")
df.to_csv("data/cleaned_customer_churn.csv", index=False)