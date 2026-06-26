# Fraud Detection using Machine Learning

A machine learning pipeline that detects fraudulent credit card transactions using the Kaggle "Credit Card Fraud Detection" dataset (284,807 transactions, 0.17% fraud).

## Project Overview
This project handles a highly imbalanced classification problem — fraud makes up less than 0.2% of all transactions. It uses SMOTE to balance the training data, then trains and compares three models.

## Pipeline
1. **Preprocessing** — scales `Amount` and `Time` features
2. **Class balancing** — SMOTE oversampling on the training set only (test set stays untouched/real-world)
3. **Model training** — Logistic Regression, Random Forest, XGBoost
4. **Evaluation** — precision, recall, F1, ROC-AUC on the held-out test set

## Results (on held-out test set, never seen during training)
Best model: **Random Forest**

| Metric | Value |
|---|---|
| Precision (fraud class) | 0.82 |
| Recall (fraud class) | 0.82 |
| F1-score (fraud class) | 0.82 |
| ROC-AUC | 0.97 |

> Note: accuracy is not a meaningful metric here since ~99.8% of transactions are legitimate — a model that predicts "not fraud" every time would score 99.8% accuracy while catching zero fraud. Precision/recall/F1 on the fraud class and ROC-AUC are the metrics that actually matter.

## Project Structure
fraud-detection/

├── data/raw/              # dataset (not tracked in git, see below)

├── src/

│   ├── data_preprocessing.py

│   ├── feature_engineering.py

│   ├── train.py

│   ├── evaluate.py

│   └── predict.py

├── reports/figures/        # confusion matrix & ROC curve outputs

├── main.py                 # runs the full pipeline

├── app.py                  # Flask API for serving predictions

└── requirements.txt
## Output

### Confusion Matrix
![Confusion Matrix](reports/figures/confusion_matrix_Best%20Model.png)

### ROC Curve
![ROC Curve](reports/figures/roc_curve_Best%20Model.png)

## How to Run
1. Clone the repo and create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/raw/`
4. Run the pipeline:
```bash
   python main.py
```
5. (Optional) Start the prediction API:
```bash
   python app.py
```

## Tech Stack
Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · matplotlib/seaborn · Flask