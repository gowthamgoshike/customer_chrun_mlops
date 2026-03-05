import pandas as pd


def load_data(path):
    df = pd.read_csv("/Users/gowthamgoshike/projects/customer_chrun_mlops/data/raw/telco.csv")
    return df


def clean_data(df):

    # convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_categorical(df):

    categorical_cols = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def save_processed_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":

    df = load_data("data/raw/telco_churn.csv")

    df = clean_data(df)

    df = encode_categorical(df)

    save_processed_data(df, "data/processed/churn_processed.csv")

    print("Data processing completed.")