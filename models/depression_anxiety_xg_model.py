import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import xgboost as xgb

MODEL_PATH = "models_saved/model_depression_anxiety_xg.pkl"

def train_model(data_path="../pre_processed/processed_depression_anxiety.csv"):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("depressiveness", axis=1)
    y = df["depressiveness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Actual model
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return y_pred, y_prob

def plot_feature_correlation(X):
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap="viridis", annot=False)
    plt.title("Feature Correlation Matrix")
    plt.show()

def predict_with_confidence(model, X, threshold=0.5):
    # Get predicted probabilities
    y_prob = model.predict_proba(X)[:, 1]

    # Get predicted classification
    predictions = (y_prob >= threshold).astype(int)

    # Confidence
    confidences = y_prob * predictions + (1 - y_prob) * (1 - predictions)

    return predictions, confidences


def main():
    model, X_test, y_test = train_model()

    y_pred, y_prob = evaluate_model(model, X_test, y_test, threshold = 0.5)

    plot_feature_correlation(X_test)

if __name__ == "__main__":
    main()