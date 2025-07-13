# credit_card_fraud_detection_v2.py

# Detecting Credit Card Fraud with Random Forest Classifier

# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# Load the Credit Card Transactions Dataset
data = pd.read_csv("creditcard.csv")

# Display Basic Information about the Dataset
print("First 5 Rows of the Dataset:\n", data.head())
print("\nStatistical Summary:\n", data.describe())

# Separate Fraudulent and Legitimate Transactions
fraudulent_transactions = data[data["Class"] == 1]
legitimate_transactions = data[data["Class"] == 0]

print(f"\nTotal Fraudulent Transactions: {len(fraudulent_transactions)}")
print(f"Total Legitimate Transactions: {len(legitimate_transactions)}")
print(f"Fraud Ratio: {len(fraudulent_transactions) / len(data):.6f}")

# Analyze Transaction Amounts
print("\nStatistics for Fraudulent Transaction Amounts:\n", fraudulent_transactions["Amount"].describe())
print("\nStatistics for Legitimate Transaction Amounts:\n", legitimate_transactions["Amount"].describe())

# Visualize Feature Correlation
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), vmax=0.8, square=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Features")
plt.show()

# Define Features and Target Variable
features = data.drop("Class", axis=1)
target = data["Class"]

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# OPTIONAL: Limit Training Data for Faster Training (comment out to use full dataset)
X_train = X_train[:5000]
y_train = y_train[:5000]

# Initialize and Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)

print("\nStarting the training process for the Random Forest model...")
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# Make Predictions and Evaluate the Model
predictions = rf_model.predict(X_test)

# Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)

print("\nEvaluation Metrics for the Model:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"MCC:       {mcc:.4f}")

# Plot the Confusion Matrix
confusion_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Legitimate", "Fraudulent"],
            yticklabels=["Legitimate", "Fraudulent"])
plt.title("Confusion Matrix Visualization")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()
