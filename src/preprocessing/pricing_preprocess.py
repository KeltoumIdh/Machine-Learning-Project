import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_pricing_data(path):
    df = pd.read_csv(path)

    # drop invalid values
    df = df[df["demand_index"] >= 0]

    # fill missing values
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop(columns=["dynamic_price"])
    y = df["dynamic_price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("FEATURE ORDER:", list(X.columns))

    return X_scaled, y.values, scaler

