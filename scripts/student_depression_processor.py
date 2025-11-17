import pandas as pd

df = pd.read_csv("student_depression_dataset.csv")

df["Profession"] = df["Profession"].apply(
    lambda x: "Unemployed" if x == "Student" else "Employed"
)

df["Sleep Duration"] = df["Sleep Duration"].apply(
    lambda x: "Poor" if x == "Less than 5 hours"
    else "Fair" if x == "5-6 hours"
    else "Good" if x == "7-8 hours"
    else "Excellent"
)

df["Degree"] = df["Degree"].apply(
    lambda x: "High School" if x == "Class 12"
    else "Bachelor's" if x.startswith('B') or x == "LLB"
    else "Master's" if x.startswith('M') or x == "LLM"
    else "PhD" if x == "PhD"
    else "Other"
)

health_as_num = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

num_as_health = {
    1: "Unhealthy",
    2: "Moderate",
    3: "Healthy"
}

boolean_map = {
    "Yes": True,
    "No": False
}

df["Dietary Habits"] = df["Dietary Habits"].map(health_as_num)
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(boolean_map)
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(boolean_map)

df["Health_Risks"] = df["Dietary Habits"] - df["Family History of Mental Illness"].astype(int) - df["Have you ever had suicidal thoughts ?"].astype(int)
df["Health_Risks"] = df["Health_Risks"].clip(lower=1, upper=3)
df = df[df["Health_Risks"].notna()]

df["Health_Risks"] = df["Health_Risks"].map(num_as_health)

df.drop(['id','City',"Dietary Habits", "Family History of Mental Illness", "Have you ever had suicidal thoughts ?"], axis=1,inplace=True)

df.rename(columns={"Profession": "Employment",
                   "Degree": "Education",
                   "Sleep Duration": "Sleep_Quality"
        },inplace=True)

first_cols = ["Gender", "Age", "Education", "Employment", "Sleep_Quality", "Health_Risks"]
other_cols = [col for col in df.columns if col not in first_cols]
new_order = first_cols + other_cols
df = df[new_order]

df.to_csv('processed_student_depression_dataset.csv',index=False)