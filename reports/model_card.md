# HealthAI Model Card

## Scope
This project provides a modular healthcare analytics demo including classification, regression, clustering, association rules, and sentiment analysis.

## Data
Synthetic demo datasets were used for tabular patient data and patient feedback text.

## Models
- RandomForestClassifier for risk prediction
- RandomForestRegressor for length of stay prediction
- KMeans for patient clustering
- Apriori association rules for comorbidity pattern discovery
- TF-IDF + Logistic Regression for sentiment analysis

## Ethics
No real patient PII is used. This is a learning/demo system and must not be used for clinical decision-making.

## Metrics
- classification_accuracy: 0.92
- classification_f1_weighted: 0.9195
- regression_mae: 1.3634
- regression_rmse: 1.6206
- regression_r2: 0.7474
- clustering_silhouette: 0.4624
- sentiment_precision_weighted: 0.1111
- sentiment_recall_weighted: 0.3333
- sentiment_f1_weighted: 0.1667
