import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

df = pd.read_csv("data/processed_customer_churn.csv")

# Feature and Target 
x = df.drop("Churn",axis=1)
y = df["Churn"]

# Train Test Split
X_test,X_train,y_test,y_train = train_test_split(x,y,test_size=0.2,random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model train
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# ROC-AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("\nROC-AUC Score:", roc_auc)


joblib.dump(model, "models/customer_churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nModel Saved Successfully!")