import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


mlflow.set_experiment("Customer_Churn_Prediction")


df = pd.read_csv("data/processed/features.csv")

y = df["Churn_Yes"]
X = df.drop(["Churn_Yes"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


with mlflow.start_run():

    n_estimators = 100

    model = RandomForestClassifier(n_estimators=n_estimators)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)

    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)