import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_preprocess_pricing_data(
    path,
    show_eda=True,
    save_clean=True
):
    """
    Professional preprocessing pipeline for Pricing Dataset
    - Handles negative values robustly
    - Visualizes BEFORE vs AFTER
    - Saves cleaned data to data/processed
    - Returns ready-to-train arrays
    """
    path = Path(path)
    processed_dir = path.parents[1] / "processed"
    processed_dir.mkdir(exist_ok=True)
    processed_path = processed_dir / "pricing_cleaned.csv"

    # 1️⃣ Load RAW data
    raw_df = pd.read_csv(path)
    df = raw_df.copy()

    # 2️⃣ EDA (RAW)
    if show_eda:
        print("\nRAW DATA OVERVIEW")
        print(raw_df.describe())
        print("\nRAW MISSING VALUES")
        print(raw_df.isna().sum())

    # 3️⃣ Target handling
    df = df.dropna(subset=["dynamic_price"])

    # 4️⃣ Non-negative columns (include all that cannot be negative)
    non_negative_cols = [
        "demand_index",
        "operational_cost",
        "marketing_intensity",
        "competition_pressure",
        "time_slot",
        "seasonality_index"
    ]
    for col in non_negative_cols:
        df.loc[df[col] < 0, col] = np.nan

    # 5️⃣ Missing value handling
    df[["demand_index", "marketing_intensity"]] = df[["demand_index", "marketing_intensity"]].interpolate(method="linear").bfill().ffill()
    df["operational_cost"] = df["operational_cost"].fillna(df["operational_cost"].median())
    df[["competition_pressure", "seasonality_index"]] = df[["competition_pressure", "seasonality_index"]].fillna(df[["competition_pressure", "seasonality_index"]].mean())
    df["time_slot"] = df["time_slot"].fillna(df["time_slot"].mode()[0])
    df["day_of_week"] = df["day_of_week"].fillna(df["day_of_week"].mode()[0])

    # Ensure integer columns remain integer type
    df["time_slot"] = df["time_slot"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)

    # 6️⃣ Outlier handling (IQR clipping, clip negative to 0)
    num_cols = df.select_dtypes(include="number").columns
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    df[num_cols] = df[num_cols].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)

    # After clipping, ensure inherently positive columns are ≥0
    df[non_negative_cols] = df[non_negative_cols].clip(lower=0)

    # 7️⃣ BEFORE vs AFTER visualization
    if show_eda:
        for col in ["demand_index", "operational_cost", "marketing_intensity", "dynamic_price"]:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].boxplot(raw_df[col].dropna())
            ax[0].set_title(f"BEFORE – {col}")
            ax[1].boxplot(df[col].dropna())
            ax[1].set_title(f"AFTER – {col}")
            plt.suptitle(f"Before vs After Cleaning ({col})")
            plt.tight_layout()
            plt.show()

    # 8️⃣ Cyclical features
    df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 28)
    df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 28)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 9️⃣ Save cleaned data
    if save_clean:
        df.to_csv(processed_path, index=False)
        print(f"✅ Cleaned data saved to: {processed_path}")

    # 10️⃣ Final features
    FEATURES = [
        "demand_index",
        "operational_cost",
        "marketing_intensity",
        "seasonality_index",
        "competition_pressure",
        "time_slot_sin",
        "time_slot_cos",
        "day_of_week_sin",
        "day_of_week_cos"
    ]
    X = df[FEATURES].values
    y = df["dynamic_price"].values
    meta = {"features": FEATURES, "processed_path": str(processed_path)}

    return X, y, meta
