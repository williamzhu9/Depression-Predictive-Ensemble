from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
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



def load_XY(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    target_col: str = TARGET_COL,
):
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



def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_XY()

    print("Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    print(f"  X_test : {X_test.shape},  y_test : {y_test.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"\nscale_pos_weight: {scale_pos_weight:.3f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"], 
        "eta": 0.03,             
        "max_depth": 3,           
        "min_child_weight": 10,   
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,       
        "reg_alpha": 0.2,        
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": 42,
        "nthread": 8,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=4000,
        early_stopping_rounds=50,
        verbose_eval=50,
        evals_result=evals_result,
    )

    y_pred_valid = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    y_pred_valid_bin = (y_pred_valid >= 0.5).astype(int)

    val_auc = roc_auc_score(y_valid, y_pred_valid)
    val_logloss = log_loss(y_valid, y_pred_valid)
    val_acc = accuracy_score(y_valid, y_pred_valid_bin)

    print("\nValidation Results:")
    print(f"  LogLoss:    {val_logloss:.4f}")
    print(f"  AUC:        {val_auc:.4f}")
    print(f"  Accuracy@0.5: {val_acc:.4f}")

    p_valid = float(y_valid.mean())
    base_ll = baseline_logloss_from_rate(p_valid)
    print(f"\nValid positive rate: {p_valid:.4f}")
    print(f"Baseline logloss (predict mean): {base_ll:.4f}")

    y_pred_test = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    y_pred_test_bin = (y_pred_test >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test_bin)

    print("\nTest Results:")
    print(f"  AUC:        {test_auc:.4f}")
    print(f"  LogLoss:    {test_logloss:.4f}")
    print(f"  Accuracy@0.5: {test_acc:.4f}")

    gain = model.get_score(importance_type="gain")
    imp = pd.Series(gain).sort_values(ascending=False)
    print("\nTop 20 feature importances (gain):")
    print(imp.head(20))

    prec, rec, thr = precision_recall_curve(y_valid, y_pred_valid)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    best_f1 = float(f1[best_idx])

    print(f"\nBest F1 threshold on valid: {best_thr:.3f}, F1={best_f1:.4f}")

    y_test_bin_tuned = (y_pred_test >= best_thr).astype(int)
    test_acc_tuned = accuracy_score(y_test, y_test_bin_tuned)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_bin_tuned).ravel()
    print(f"Test Accuracy (tuned threshold): {test_acc_tuned:.4f}")
    print(f"Confusion Matrix @tuned threshold: TN={tn} FP={fp} FN={fn} TP={tp}")



if __name__ == "__main__":
    main()
