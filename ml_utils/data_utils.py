"""
===========================================
GENERIC DATA CLEANING & ANALYSIS MODULE
===========================================

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# 1. LOAD DATA
# ============================

def load_data(path):
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape}")
    return df


# ============================
# 2. BASIC DATA ANALYSIS
# ============================

def analyze_data(df):
    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE ---")
    print(df.describe(include="all"))

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())


# ============================
# 3. DATA CLEANING
# ============================

def clean_data(df):
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


# ============================
# 4. ENCODING
# ============================

def encode_categorical(df):
    df = pd.get_dummies(df, drop_first=True)
    return df


# ============================
# 5. OUTLIER HANDLING (OPTIONAL)
# ============================

def remove_outliers(df, factor=1.5):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


# ============================
# 6. VISUALIZATION
# ============================

def plot_distribution(df, column):
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
