import pandas as pd

df = pd.read_csv("../raw/student_depression_dataset.csv")

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
    "No": 0,
}

gender_map = {
    "Male": 1,
    "Female": 0
}

def profession_simplification(x):
    if isinstance(x, str):
        x_lower = x.lower()
        
        # Student group
        if "student" in x_lower:
            return "student"
        
        # Unemployed group
        if "unemployed" in x_lower or "none" in x_lower or "other" in x_lower:
            return "unemployed"
        
        return "employed"
    
profession_map = {
    "student": 0,
    "unemployed": 0,
    "employed": 1
}

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

# Treating 'other' as basic education
degree_multiclass ={
    "high school": 0,
    "other": 0,
    "bachelor's": 1,
    "master's": 2,
    "phd": 3,
}

# Drop all missing data rows
df.dropna(inplace=True)
# Treating 'other' as missing data to reduce noise for sleep duration
df = df[~df['Sleep Duration'].isin(['Others'])]
# Dropping id and city to reduce noise and prevent overfitting (we don't care about id and we don't want city to influence classification due to lack of city data)
df.drop(['id', "City"], axis=1,inplace=True)

# Map all columns to proper encoding
df["Dietary Habits"] = df["Dietary Habits"].map(health_multiclass)
df["Sleep Duration"] = df["Sleep Duration"].map(sleep_multiclass)
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(boolean_map)
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(boolean_map)
df["Degree"] = df["Degree"].apply(degree_map).map(degree_multiclass)
df["Gender"] = df["Gender"].map(gender_map)
df["Profession"] = df["Profession"].apply(profession_simplification).map(profession_map)

# Rename degree to education level
df.rename(columns={"Profession": "Employment", "Degree": "Education Level"},inplace=True)
df.columns = df.columns.str.lower()

# Reorder columns so that T/F are together and Numerics are together
last_cols = ["employment", "have you ever had suicidal thoughts ?", "family history of mental illness", "depression"]
other_cols = [col for col in df.columns if col not in last_cols]
new_order = other_cols + last_cols
df = df[new_order]

# Write processed data to csv
df.to_csv("../pre_processed/processed_student_depression.csv",index=False)
print("Write successful to pre_processed/processed_student_depression.csv")