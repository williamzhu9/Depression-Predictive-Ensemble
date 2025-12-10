import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import xgboost as xgb

def train_model(data_path="../pre_processed/processed_student_depression.csv"):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("depression", axis=1)
    y = df["depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

def main():
    model, X_test, y_test = train_model()

    y_pred, y_prob = evaluate_model(model, X_test, y_test, threshold = 0.5)

    plot_feature_correlation(X_test)

if __name__ == "__main__":
    main()