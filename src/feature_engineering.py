import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def create_features(df):
    df = df.copy()
    df['hour'] = (df['Time_scaled'] * 24).astype(int) % 24
    df['amount_log'] = np.log1p(df['Amount_scaled'].abs())
    return df

def handle_imbalance(X_train, y_train):
    print(f"Before SMOTE - Fraud: {y_train.sum()} | Normal: {(y_train==0).sum()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE  - Fraud: {y_resampled.sum()} | Normal: {(y_resampled==0).sum()}")
    return X_resampled, y_resampled