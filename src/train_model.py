import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    data_path = "data/processed/features.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run feature_engineering.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 1. Define X and y
    # Because we used drop_first=True in the previous script, 'Churn' automatically became 'Churn_Yes'
    y = df["Churn_Yes"]
    X = df.drop(["Churn_Yes"], axis=1)

    # 2. Train/Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Initialize and Train the Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate the Model
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)

    print("\n" + "="*30)
    print(f"Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {roc:.4f}")
    print("="*30 + "\n")

    # 5. Save the trained model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/churn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Success! Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()