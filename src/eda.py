import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data\cleaned_customer_churn.csv')
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Shape:")
print(df.shape)
print("\nStatistical Summary:")
print(df.describe())

# Target Variable Analysis
print("\nChurn Count:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100) # Why normalize=True? without it gives count with it gives gives proportions.Multiplying by 100 converts proportions into percentages.

#visualize the target variable:
sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("images/churn_distribution.png")
plt.show()

# Gender vs Churn
plt.figure(figsize=(6,4))

sns.countplot(data=df, x="gender", hue="Churn")

plt.title("Gender vs Customer Churn")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("images/gender_vs_churn.png")
plt.show()

# Contract vs Churn
plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("images/contract_vs_churn.png")
plt.show()

# Internet Service vs Churn
plt.figure(figsize=(8,5))

sns.countplot(data=df, x="InternetService", hue="Churn")

plt.title("Internet Service vs Customer Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("images/internet_service_vs_churn.png")
plt.show()

# Payment Method vs Churn
plt.figure(figsize=(10,5))

sns.countplot(data=df, x="PaymentMethod", hue="Churn")

plt.title("Payment Method vs Customer Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("images/payment_method_vs_churn.png")
plt.show()

# Partner vs Churn
plt.figure(figsize=(6,4))

sns.countplot(data=df, x="Partner", hue="Churn")

plt.title("Partner vs Customer Churn")
plt.xlabel("Partner")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("images/partner_vs_churn.png")
plt.show()

# Dependents vs Churn
plt.figure(figsize=(6,4))

sns.countplot(data=df, x="Dependents", hue="Churn")

plt.title("Dependents vs Customer Churn")
plt.xlabel("Dependents")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("images/dependents_vs_churn.png")
plt.show()