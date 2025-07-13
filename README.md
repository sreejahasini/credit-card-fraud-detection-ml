# credit-card-fraud-detection-ml
Built a supervised ML model to detect fraudulent transactions in real-world, imbalanced data. Includes EDA, correlation analysis, and model evaluation using precision, recall, F1-score, and MCC. Great for hands-on experience with classification in financial datasets.
💳 Credit Card Fraud Detection
This project is a simple machine learning model that detects fraudulent credit card transactions using the Credit Card Fraud Detection dataset from Kaggle.

It uses a Random Forest Classifier to distinguish between fraudulent and legitimate transactions based on anonymized features.

📂 Dataset Source
The dataset can be downloaded from Kaggle:
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Note: Due to licensing restrictions, the actual CSV file is not included in this repository. You need to download it manually and place it in the same directory as the script.

🧠 Technologies & Libraries Used
Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

📊 Features
Loads and explores the dataset.

Detects class imbalance (fraud vs. legit).

Visualizes correlations using a heatmap.

Trains a RandomForestClassifier on the data.

Evaluates the model with:

Accuracy

Precision

Recall

F1 Score

Matthews Correlation Coefficient

Plots the confusion matrix.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset from Kaggle and place creditcard.csv in the same folder.

Run the script:

bash
Copy
Edit
python credit_card_fraud_detection.py
📌 Sample Output
Model Evaluation Metrics

Heatmap of Correlations

Confusion Matrix Visualization

📄 License
This project is open-source and free to use. No license is applied, but feel free to add MIT License if you'd like others to use it freely.

🙋‍♀️ Author
Created by U.V.Sreeja Hasini – Feel free to connect on GitHub or LinkedIn!
