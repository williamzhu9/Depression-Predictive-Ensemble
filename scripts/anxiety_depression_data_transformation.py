import pandas as pd

#importing file 
csv_path = "../raw/anxiety_depression_data.csv"
df = pd.read_csv(csv_path)

def transform_medication_use(medication_use):
    if medication_use == "Occasional":
        return 1
    elif medication_use == "Regular":
        return 2
    else:
        return 0

def transform_substance_use(substance_use):
    if substance_use == "Occasional":
        return 1
    elif substance_use == "Frequent":
        return 2
    else:
        return 0

def transform_education(education):
    if education == "High School":
        return 0
    elif education == "Bachelor's":
        return 1
    elif education == "Master's":
        return 2
    elif education == "PhD":
        return 3
    elif education == "Other":
        return 4
    else:
        return "FAILED"

def is_depressed(depression_score):
    if depression_score >= 11:
        return 1
    else:
        return 0

df["Medication_Use"] = df["Medication_Use"].apply(transform_medication_use)
df["Substance_Use"] = df["Substance_Use"].apply(transform_substance_use)
df["Education_Level"] = df["Education_Level"].apply(transform_education)
df["Depression_Score"] = df["Depression_Score"].apply(is_depressed)

df.rename(columns={"Depression_Score": "is_depressed"},inplace=True)

df = pd.get_dummies(df, columns=['Employment_Status', 'Gender'])
df = df.astype(int)
df.columns = df.columns.str.lower()

df.to_csv("../staging/processed_anxiety_depression_data.csv", index=False)
print("Write successful")