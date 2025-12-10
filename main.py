# main.py
import pandas as pd
from ensemble import weighted_vote

# -----------------------------
# Step 1: Load models
# -----------------------------
from train_models.models.model_a import get_model as load_model_a
from train_models.models.models.model_b import get_model as load_model_b
from train_models.models.models.model_c import get_model as load_model_c

# Load trained models
models = [
    load_model_a(),
    load_model_b(),
    load_model_c(),
]

# -----------------------------
# Step 2: Define feature groups for each model
# -----------------------------
# TODO: fill in the features each model expects
feature_groups = [
    [],  # features for model_a
    [],  # features for model_b
    [],  # features for model_c
]

# Optional: assign weights to each model for voting
weights = [
    1.0,  # weight for model_a
    1.0,  # weight for model_b
    1.0,  # weight for model_c
]

# -----------------------------
# Step 3: Load input CSV
# -----------------------------
input_df = pd.read_csv("input_records.csv")  # your new data
predictions = []

# -----------------------------
# Step 4: Make ensemble predictions
# -----------------------------
for idx, record in input_df.iterrows():
    final_class, final_prob = weighted_vote(models, feature_groups, record, weights)
    predictions.append({
        "index": idx,
        "final_class": final_class,
        "final_prob": final_prob
    })

# -----------------------------
# Step 5: Output results
# -----------------------------
pred_df = pd.DataFrame(predictions)
print(pred_df)

# Optional: save to CSV
pred_df.to_csv("ensemble_predictions.csv", index=False)
