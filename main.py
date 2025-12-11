# main.py
import pandas as pd
import joblib
import os
from scripts.student_depression_processor import preprocess_student_depression
from scripts.depression_anxiety_processor import preprocess_depression_anxiety

MODEL_DIR = "models/models_saved"

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path)
    
models = {
    "da_rf": load_model("model_depression_anxiety_rf.pkl"),
    # "da_xg": load_model("model_depression_anxiety_xg.pkl"),
    "sd_rf": load_model("model_student_depression_rf.pkl"),
    "sd_xg": load_model("model_student_depression_xg.pkl")
}

# Processed data
feature_groups = [
    # features in depression_anxiety
    [
        "school_year",
        "age",
        "gender",
        "bmi",
        "who_bmi",
        "phq_scores",
        "depression_severity",
        "gad_score",
        "anxiety_severity",
        "epworth_score",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatmnet",
        "sleepiness"
    ],  
    # features in student_depression
    [      
        "gender",
        "age",
        "academic pressure",
        "work pressure",
        "cgpa",
        "study satisfaction",
        "job satisfaction",
        "sleep duration",
        "dietary habits",
        "education level",
        "work/study hours",
        "financial stress",
        "employment",
        "have you ever had suicidal thoughts ?",
        "family history of mental illness"
    ],  
]

# Raw data
raw_columns = [
    # Raw columns in depression_anxiety
    [
        "school_year",
        "age",
        "gender",
        "bmi",
        "who_bmi",
        "phq_score",
        "depression_severity",
        "suicidal",
        "depression_diagnosis",
        "depression_treatment",
        "gad_score",
        "anxiety_severity",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatment",
        "epworth_score",
        "sleepiness"
    ],  
    # Raw columns in student_depression
    [      
        "gender",
        "age",
        "academic pressure",
        "work pressure",
        "cgpa",
        "study satisfaction",
        "job satisfaction",
        "sleep duration",
        "dietary habits",
        "degree",
        "work/study hours",
        "financial stress",
        "profession",
        "have you ever had suicidal thoughts ?",
        "family history of mental illness"
    ],  
]

# Load input
input_df = pd.read_csv("raw/input/input.csv")
input_df.columns = input_df.columns.str.lower()

model_inputs = {}
processed_inputs = {}
ensemble_preds = {}

for i, features in enumerate(raw_columns):
    # Keep only columns that exist in input_df
    cols_to_use = [c for c in features if c in input_df.columns]
    
    model_inputs[f"dataset{i}"] = input_df[cols_to_use].copy()

for key, df in model_inputs.items():
    if key == 'dataset0':
        processed_inputs[key] = preprocess_depression_anxiety(df)
    if key == 'dataset1':
        processed_inputs[key] = preprocess_student_depression(df)

# map models to processed datasets
model_to_data = {
    "da_rf": processed_inputs["dataset0"],
    "da_xg": processed_inputs["dataset0"],
    "sd_rf": processed_inputs["dataset1"],
    "sd_xg": processed_inputs["dataset1"]
}

for name, model in models.items():
    df_proc = model_to_data[name]  # select the correct processed dataset
    
    preds = model.predict(df_proc)
    probs = model.predict_proba(df_proc).max(axis=1)  # confidence per row

    pred_df = df_proc.copy()
    pred_df["pred_class"] = preds
    pred_df["pred_confidence"] = probs

    # Save predictions in a dictionary
    ensemble_preds[name] = pred_df

# Print and save predictions per model
for name, pred_df in ensemble_preds.items():
    print(f"Predictions from {name} model:")
    print(pred_df.head())
    pred_df.to_csv(f"ensemble_predictions_{name}.csv", index=False)
