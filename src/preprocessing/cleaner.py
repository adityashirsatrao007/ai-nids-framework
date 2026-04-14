import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the CICIDS2017 dataframe:
    - Strips whitespace from column names
    - Replaces inf/-inf with NaN
    - Drops rows with NaN
    - Ensures numeric columns are correct
    """
    print("Pre-cleaning: Columns stripping and replacing inf...")
    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    
    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for NaN counts
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} missing/inf values. Dropping rows...")
        df.dropna(inplace=True)
    
    # Standard CICIDS2017 cleaning: Some files have "Benign" as "BENIGN"
    if 'Label' in df.columns:
        df['Label'] = df['Label'].str.upper().str.strip()
        
    return df

def basic_preprocessing(df):
    """
    Basic encoding and preparing for scaling.
    """
    # Identify non-numeric columns besides 'Label'
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Convert 'Label' to numeric (Binary for now, can be changed to multiclass)
    # We will use label encoding for the target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le
