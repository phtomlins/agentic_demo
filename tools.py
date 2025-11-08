# tools.py
import pandas as pd
from sklearn.utils import resample

def load_dataset(path):
    return pd.read_csv(path)

def check_schema(df):
    return df.dtypes.to_string()

def check_class_imbalance(df, target_col):
    return df[target_col].value_counts(normalize=True).to_string()

def recommend_preprocessing(df):
    recommendations = []
    if df.isnull().sum().any():
        recommendations.append("Impute missing values.")
    if any(df.dtypes == 'object'):
        recommendations.append("Encode categorical variables.")
    return "\n".join(recommendations)