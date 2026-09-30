# AI-Based Financial Fraud Detection

## About the Project

My AI-Based Financial Fraud Detection project is a machine-learning system designed to identify potentially fraudulent transactions by analyzing transaction patterns and account history.

I used the IEEE-CIS Fraud Detection dataset and created transaction sequences using sliding windows so the models could learn behavioral patterns over time.

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* SNN / snntorch
* Flask
* Pandas
* NumPy

## Models Used

* Mamba-KAN — for learning patterns from transaction sequences
* Spiking Neural Network — for temporal/spike-based patterns
* Logistic Regression — baseline model
* Random Forest — baseline model

## Model Evaluation

I evaluated the models using:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Random Forest performed better overall on the real test data, while the sequence models showed strong fraud recall.

## Additional Features

We also added:

* **Cold-start handling** — handles accounts with very little transaction history.
* **Fraud-ring detection** — uses a GNN to identify suspicious connections between accounts through shared devices.
* **Concept-drift detection** — checks whether fraud patterns change over time.
* **Risk scoring** — classifies transactions as LOW, MEDIUM, or HIGH risk.
* **Explainability** — shows which features contributed to a prediction.

## Dashboard

The project is integrated with a Flask dashboard where users can inspect transactions, view model predictions, risk levels, evaluation results, and feature explanations.

## What I Learned

Through this project, I learned about data preprocessing, feature engineering, sequential modelling, class imbalance, model evaluation, fraud detection, and applying machine learning to real-world problems.
