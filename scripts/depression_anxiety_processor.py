import pandas as pd
import os

# Mappings
gender_map = {
    "male": 1,
    "female": 0
}

who_bmi_map = {
    "Underweight": 0,
    "Normal": 1,
    "Overweight": 2,
    "Class I Obesity": 3,
    "Class II Obesity": 4,
    "Class III Obesity": 5
}

severity_map = {
    "None-minimal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Moderately severe": 3,
    "Severe": 4
}

boolean_map = {
    True: 1,
    False: 0
}

def preprocess_depression_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    
    df = df.dropna(axis=0)

    # Dropping these because of leakage
    drop_cols = [col for col in [
        'id',
        'depression_diagnosis', 
        'depression_treatment', 
        'depression_severity', 
        'suicidal'
    ] if col in df.columns]

    df = df.drop(drop_cols, axis=1)
    df = df[~df['who_bmi'].isin(['Not Availble'])]
    
    df['gender'] = df['gender'].str.lower().map(gender_map)
    df['who_bmi'] = df['who_bmi'].map(who_bmi_map)
    df['anxiety_severity'] = df['anxiety_severity'].map(severity_map)
    
    # Map boolean columns
    boolean_cols = ['depressiveness', 'sleepiness', 'anxiousness', 'anxiety_diagnosis', 'anxiety_treatment']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].map(boolean_map)
    
    last_cols = [col for col in boolean_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in last_cols]
    df = df[other_cols + last_cols]

    return df

if __name__ == "__main__":
    raw_path = "../raw/training/depression_anxiety_dataset.csv"
    processed_path = "../pre_processed/processed_depression_anxiety.csv"
    
    if not os.path.exists(os.path.dirname(processed_path)):
        os.makedirs(os.path.dirname(processed_path))
    
    df_processed = preprocess_depression_anxiety(pd.read_csv(raw_path))
    df_processed.to_csv(processed_path, index=False)
    print(f"Write successful to {processed_path}")
