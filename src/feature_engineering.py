import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

def create_features(df):
    # 1. Drop customerID so we don't accidentally create 7000+ dummy columns!
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. Create the tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
    )

    # 3. Average spend per month
    df["avg_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # 4. ENCODE ALL STRING COLUMNS AT ONCE
    # By not specifying 'columns', it automatically converts all text columns 
    # (gender, Contract, Churn, tenure_group, etc.) into numeric 1s and 0s.
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    return df

def save_features(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Feature engineering completed. Saved to {path}")

if __name__ == "__main__":
    processed_path = "data/processed/churn_processed.csv"
    feature_path = "data/processed/features.csv"

    df = load_data(processed_path)
    df = create_features(df)
    save_features(df, feature_path)