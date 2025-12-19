import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_credit_data(path):
    df = pd.read_csv(path)

    df = df.dropna(subset=["risk_level"])
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["risk_level"])
    y = df["risk_level"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X.values, y_encoded, encoder
