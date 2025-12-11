import pandas as pd
import os

health_multiclass = {
    "Unhealthy": 0, 
    "Moderate": 1, 
    "Healthy": 2
}

sleep_multiclass = {
    "Less than 5 hours": 0, 
    "5-6 hours": 1, 
    "7-8 hours": 2, 
    "More than 8 hours": 3
}

boolean_map = {
    "Yes": 1, 
    "No": 0
}

gender_map = {
    "Male": 1, 
    "Female": 0
}

profession_map = {
    "student": 0, 
    "unemployed": 0, 
    "employed": 1
}

degree_multiclass = {
    "high school": 0, 
    "other": 0, 
    "bachelor's": 1, 
    "master's": 2, 
    "phd": 3
}

def profession_simplification(x):
    if isinstance(x, str):
        x_lower = x.lower()
        if "student" in x_lower:
            return "student"
        if "unemployed" in x_lower or "none" in x_lower or "other" in x_lower:
            return "unemployed"
        return "employed"
    return x

def degree_map(x):
    if x == "Class 12":
        return "high school"
    elif x.startswith("B") or x == "LLB":
        return "bachelor's"
    elif x.startswith("M") or x == "LLM":
        return "master's"
    elif x == "PhD":
        return "phd"
    else:
        return "other"

def preprocess_student_depression(df: pd.DataFrame) -> pd.DataFrame:
    # Drop missing or useless data
    df.columns = df.columns.str.lower()

    df = df[~df['sleep duration'].isin(['Others'])]
    drop_cols = [col for col in ['id', 'city'] if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    df["dietary habits"] = df["dietary habits"].map(health_multiclass)
    df["sleep duration"] = df["sleep duration"].map(sleep_multiclass)
    df["family history of mental illness"] = df["family history of mental illness"].map(boolean_map)
    df["have you ever had suicidal thoughts ?"] = df["have you ever had suicidal thoughts ?"].map(boolean_map)
    df["degree"] = df["degree"].apply(degree_map).map(degree_multiclass)
    df["gender"] = df["gender"].map(gender_map)
    df["profession"] = df["profession"].apply(profession_simplification).map(profession_map)
    
    # Rename columns
    df.rename(columns={"profession": "employment", "degree": "education level"}, inplace=True)
    
    # Group booleans at the end
    last_cols = ["employment", "have you ever had suicidal thoughts ?", "family history of mental illness"]
    other_cols = [col for col in df.columns if col not in last_cols]
    df = df[other_cols + last_cols]

    return df

if __name__ == "__main__":
    raw_path = "../raw/training/student_depression_dataset.csv"
    processed_path = "../pre_processed/processed_student_depression.csv"
    
    if not os.path.exists(os.path.dirname(processed_path)):
        os.makedirs(os.path.dirname(processed_path))
    
    df_processed = preprocess_student_depression(pd.read_csv(raw_path))
    df_processed.to_csv(processed_path, index=False)
    print(f"Write successful to {processed_path}")
