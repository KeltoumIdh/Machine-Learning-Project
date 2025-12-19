import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_pricing_data(path, show_eda=True):
    """
    Preprocessing + optional EDA for Pricing Dataset
    - Data profiling
    - Cleaning
    - Cyclical feature engineering
    - NO scaling (done in training)
    """

    df = pd.read_csv(path)

    # -------------------------
    # 1. EDA
    # -------------------------
    if show_eda:
        print(df.info())
        print(df.describe())
        print(df.isna().sum())
        print("Duplicates:", df.duplicated().sum())

    # -------------------------
    # 2. Handle target
    # -------------------------
    df = df.dropna(subset=["dynamic_price"])

    # -------------------------
    # 3. Interpolation (high / medium variance)
    # -------------------------
    cols_interpolate = ["demand_index", "marketing_intensity"]
    df[cols_interpolate] = (
        df[cols_interpolate]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    # -------------------------
    # 4. Ordinal features
    # -------------------------
    df["time_slot"] = df["time_slot"].ffill().bfill()
    df["day_of_week"] = df["day_of_week"].fillna(
        df["day_of_week"].mode()[0]
    )

    # -------------------------
    # 5. Low variance continuous
    # -------------------------
    df[["competition_pressure", "seasonality_index"]] = (
        df[["competition_pressure", "seasonality_index"]]
        .fillna(df[["competition_pressure", "seasonality_index"]].mean())
    )

    # -------------------------
    # 6. Robust to outliers
    # -------------------------
    df["operational_cost"] = df["operational_cost"].fillna(
        df["operational_cost"].median()
    )

    # -------------------------
    # 7. Outlier visualization (EDA only)
    # -------------------------
    if show_eda:
        for col in df.select_dtypes(include="number").columns:
            plt.figure()
            df.boxplot(column=col)
            plt.title(f"Boxplot of {col}")
            plt.show()

    # -------------------------
    # 8. Cyclical feature engineering
    # -------------------------
    df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 28)
    df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 28)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    FEATURES_EXT = [
        "demand_index", "operational_cost",
        "marketing_intensity", "seasonality_index", "competition_pressure",
        "time_slot_sin", "time_slot_cos",
        "day_of_week_sin", "day_of_week_cos"
    ]

    X = df[FEATURES_EXT].values
    y = df["dynamic_price"].values

    meta = {
        "features": FEATURES_EXT
    }

    return X, y, meta
