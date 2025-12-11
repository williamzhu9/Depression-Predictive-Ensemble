from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

MODEL_PATH = "models_saved/model_depression_anxiety_rf.pkl"

# Actually train the model
def train_model(data_path="../pre_processed/processed_depression_anxiety.csv"):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("depressiveness", axis=1)
    y = df["depressiveness"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Actual model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Model saved to {MODEL_PATH}")

    return model, X_test, y_test

# Model accuracy
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return y_pred

def plot_feature_correlation(X):
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap="viridis", annot=False)
    plt.title("Feature Correlation Matrix")
    plt.show()

def predict_with_confidence(model, X):
    # Get predicted classification
    predictions = model.predict(X)
    
    # Get predicted probabilities
    proba = model.predict_proba(X)
    
    # Confidence score
    confidences = [proba[i, pred] for i, pred in enumerate(predictions)]
    
    return predictions, confidences

def main():
    model, X_test, y_test = train_model()

    y_pred = evaluate_model(model, X_test, y_test)

    plot_feature_correlation(X_test)

if __name__ == "__main__":
    main()
