import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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
    Fix data types and fill missing values properly
    """

    for col in df.columns:

        # Treat all text-like columns as categorical
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    return X_scaled


def preprocess_data(path, target_column='EngagementLevel', test_size=0.2):

    # Load data
    df = load_data(path)

    # Drop unnecessary columns
    df = drop_irrelevant_features(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Feature engineering
    df = feature_engineering(df)

    # Check target column exists
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in dataset")

    # Split BEFORE encoding/outliers (VERY IMPORTANT)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Encode categorical using get_dummies (SAFE)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Align columns (IMPORTANT)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Remove outliers ONLY from training data
    X_train = remove_outliers(X_train, numerical_cols)

    # Also adjust y_train accordingly
    y_train = y_train.loc[X_train.index]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # Handle class imbalance (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

    return X_train_final, X_test_scaled, y_train_final, y_test