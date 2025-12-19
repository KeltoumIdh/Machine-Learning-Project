import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_preprocess_credit_data(path, handle_outliers=True):
    # Load dataset
    df = pd.read_csv(path)
    print("Initial data shape:", df.shape)

    # Drop rows where target is missing
    df = df.dropna(subset=["risk_level"])

    # Separate features and target
    X = df.drop(columns=["risk_level"])
    y = df["risk_level"]

    # Handle missing values
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            # Numeric columns: fill with median
            # SimpleImputer with strategy='median' replaces missing values in the column with the median of the non-missing values.
            imputer = SimpleImputer(strategy='median')
            # This line fills missing values in the numeric column 'col' with the median of the column.
            # fit_transform() returns a 2D array, so we flatten it with .ravel() before assignment.
            X[col] = imputer.fit_transform(X[[col]]).ravel()
        else:
            # Categorical columns: fill with most frequent
            imputer = SimpleImputer(strategy='most_frequent')
            X[col] = imputer.fit_transform(X[[col]])

    # Outliers are capped (not dropped) using the IQR method.
    if handle_outliers:
        # Loop through all numeric columns to cap (not remove) outliers using the IQR method
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)           # First quartile (25th percentile)
            Q3 = X[col].quantile(0.75)           # Third quartile (75th percentile)
            IQR = Q3 - Q1                        # Interquartile range
            lower = Q1 - 1.5 * IQR               # Lower bound for outliers
            upper = Q3 + 1.5 * IQR               # Upper bound for outliers
            # Cap any value below 'lower' bound to 'lower', else keep the original value
            X[col] = np.where(X[col] < lower, lower, X[col])
            # Cap any value above 'upper' bound to 'upper', else keep the original value
            X[col] = np.where(X[col] > upper, upper, X[col])

    # Encode categorical features
    # Identify all categorical columns (type 'object') in the features dataframe X
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # For each categorical column, create a new LabelEncoder instance
        le = LabelEncoder()
        # Fit the encoder on the current column and transform the values to integer codes
        # This replaces the original string categories with numeric labels
        X[col] = le.fit_transform(X[col])

    # Encode the target variable
    # Create a LabelEncoder for the target
    target_encoder = LabelEncoder()
    # Fit and transform the target variable 'y', producing integer labels
    y_encoded = target_encoder.fit_transform(y)

    # Print out the number of missing values in each column after preprocessing, for diagnostics
    print("After preprocessing, missing values per column:\n", X.isna().sum())
    # Print the shape (rows, columns) of the processed features
    print("Processed data shape:", X.shape)

    # Return the processed feature matrix (as a numpy array), the encoded target, and the target encoder itself
    return X.values, y_encoded, target_encoder

# Example usage
# X, y, encoder = load_and_preprocess_credit_data("C:/workspace/genAi_&_ML/geeks_projects/machine_learning_project/data/raw/part3_credit_risk_dirty.csv")
