# Sample src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def handle_missing_values(df):
    # Simple strategy (you can improve if needed)
    df = df.dropna()
    return df


def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df


def feature_engineering(df):
    # Example feature (modify based on dataset)
    if 'Hours_Played' in df.columns and 'Sessions_Per_Week' in df.columns:
        df['Engagement_Score'] = df['Hours_Played'] * df['Sessions_Per_Week']
    return df


def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def full_preprocessing_pipeline(file_path, target_column, scale=True):
    # Load
    df = load_data(file_path)

    # Preprocessing
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = feature_engineering(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # SMOTE (only train)
    X_train, y_train = apply_smote(X_train, y_train)

    # Scaling (optional)
    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test, df


# How to Use in Notebooks
# At the top of each notebook:

# import sys
# sys.path.append('../src')

# from preprocessing import full_preprocessing_pipeline

# Example Usage (IMPORTANT)
# X_train, X_test, y_train, y_test, df = full_preprocessing_pipeline(
#     file_path="../data/online_gaming_behavior_dataset.csv",
#     target_column="your_target_column_name",
#     scale=True   # True for LR & SVM
# )