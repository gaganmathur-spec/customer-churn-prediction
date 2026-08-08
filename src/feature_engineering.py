import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# Find categorical columns
categorical_columns = df.select_dtypes(include="object").columns

# Dictionary to store encoders
encoders = {}

# Encode categorical columns
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# Save processed dataset
df.to_csv("data/processed_customer_churn.csv", index=False)

# Save all encoders
joblib.dump(encoders, "models/label_encoders.pkl")

print("Feature Engineering Completed!")
print("Processed dataset saved.")
print("Label Encoders saved.")