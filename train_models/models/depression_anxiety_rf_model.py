from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TARGET_COL = "depression_diagnosis"
TRAIN_PATH = PROJECT_ROOT/"pre_processed"/"depression_anxiety_train.csv"
TEST_PATH = PROJECT_ROOT/"pre_processed"/"depression_anxiety_test.csv"


# Load the training & testing data
def load_XY(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH, target_col: str = TARGET_COL):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train_full = df_train[target_col].astype("int8")
    X_train_full = df_train.drop(columns=[target_col])

    y_test = df_test[target_col].astype("int8")
    X_test = df_test.drop(columns=[target_col])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")
    X_test = X_test.astype("float32")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def baseline_logloss_from_rate(p: float) -> float:
    eps = 1e-12
    return -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))


# Actually train the model
def train_model(X_train, y_train):
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    class_weight = {0: 1, 1: neg / pos} if pos > 0 else "balanced"
    print(f"\nClass weight for positive class: {class_weight}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    return model

# Model accuracy
def evaluate_model(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    print(f"Accuracy@{threshold}: {acc:.4f}")
    return y_pred, y_prob


def feature_importance(model, X_train, top_n=20):
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"\nTop {top_n} feature importances:")
    print(imp.head(top_n))
    return imp

# Get best threshold settings for accuracy
def tune_threshold(y_valid, y_prob_valid):
    prec, rec, thr = precision_recall_curve(y_valid, y_prob_valid)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    best_f1 = float(f1[best_idx])
    print(f"\nBest F1 threshold on valid: {best_thr:.3f}, F1={best_f1:.4f}")
    return best_thr


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_XY()

    print("Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    print(f"  X_test : {X_test.shape},  y_test : {y_test.shape}")

    model = train_model(X_train, y_train)

    y_pred_valid, y_prob_valid = evaluate_model(model, X_valid, y_valid)

    # Feature importances
    feature_importance(model, X_train)

    # Tune threshold using validation
    best_thr = tune_threshold(y_valid, y_prob_valid)

    print("\n--- Test Results @0.5 threshold ---")
    y_pred_test, y_prob_test = evaluate_model(model, X_test, y_test, threshold=0.5)

    # Evaluate on test set with tuned threshold
    y_pred_test_tuned = (y_prob_test >= best_thr).astype(int)
    test_acc_tuned = accuracy_score(y_test, y_pred_test_tuned)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_tuned).ravel()
    print(f"Test Accuracy (tuned threshold): {test_acc_tuned:.4f}")
    print(f"Confusion Matrix @tuned threshold: TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
