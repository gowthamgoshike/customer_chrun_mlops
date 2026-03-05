from fastapi import FastAPI
import pandas as pd
#import mlflow.pyfunc
import pickle
app = FastAPI(title="Customer Churn Prediction API")

# Load MLflow model
#model = mlflow.pyfunc.load_model("runs:/19d8e7f4ae0448c78e8b6d5ca9c60211/model")

with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running"}


@app.post("/predict")
def predict(data: dict):
    
    # 1. THE MISSING LINK: Convert the incoming dictionary into a Pandas DataFrame
    df = pd.DataFrame([data])

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
    )
    df["avg_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # 2. Drop customerID if it's in the payload
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 3. One-hot encode the input row
    df = pd.get_dummies(df)

    # 4. THE MAGIC FIX: Align the columns
    # Get the exact list of columns the model memorized during training
    # (Checking both Scikit-Learn and MLflow formats)
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
    else:
        expected_cols = model._model_impl.sklearn_model.feature_names_in_

    # Force the dataframe to match, adding missing columns as 0s
    df = df.reindex(columns=expected_cols, fill_value=0)

    # 5. Now it's safe to predict!
    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }