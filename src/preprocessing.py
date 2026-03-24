import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset from CSV
    """
    df = pd.read_csv(path)
    return df


def drop_irrelevant_features(df):
    """
    Remove unnecessary columns
    """
    if 'PlayerID' in df.columns:
        df = df.drop(columns=['PlayerID'])
    return df


def handle_missing_values(df):
    """
    Fill missing values
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_outliers(df, numerical_cols):
    """
    Remove outliers using IQR method
    """
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def encode_categorical(df):
    """
    Convert categorical features to numeric
    """
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df


def feature_engineering(df):
    """
    Create new useful features
    """
    # Example feature: total weekly playtime
    if 'AvgSessionDurationMinutes' in df.columns and 'SessionsPerWeek' in df.columns:
        df['TotalWeeklyPlayTime'] = (
            df['AvgSessionDurationMinutes'] * df['SessionsPerWeek']
        )

    return df


def scale_features(X):
    """
    Standardize numerical features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def preprocess_data(path, target_column='EngagementLevel', test_size=0.2):
    """
    Full preprocessing pipeline
    """

    # Load data
    df = load_data(path)

    # Drop unnecessary columns
    df = drop_irrelevant_features(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Feature engineering
    df = feature_engineering(df)

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove target column from numerical list if present
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    # Remove outliers
    df = remove_outliers(df, numerical_cols)

    # Encode categorical
    df = encode_categorical(df)

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scale features
    X = scale_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test