import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

def save_processed_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

if __name__ == "__main__":
    raw_path = "data/raw/telco_churn.csv"
    processed_path = "data/processed/churn_processed.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    save_processed_data(df, processed_path)
    print("Data processing completed successfully!")