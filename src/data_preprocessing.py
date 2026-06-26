import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df

def scale_features(df, amount_scaler=None, time_scaler=None):
    df = df.copy()
    if amount_scaler is None:
        amount_scaler = StandardScaler()
        df['Amount_scaled'] = amount_scaler.fit_transform(df[['Amount']])
    else:
        df['Amount_scaled'] = amount_scaler.transform(df[['Amount']])

    if time_scaler is None:
        time_scaler = StandardScaler()
        df['Time_scaled'] = time_scaler.fit_transform(df[['Time']])
    else:
        df['Time_scaled'] = time_scaler.transform(df[['Time']])

    df.drop(columns=['Amount', 'Time'], inplace=True)
    return df, amount_scaler, time_scaler

def split_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)