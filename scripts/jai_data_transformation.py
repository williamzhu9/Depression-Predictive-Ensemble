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
        return 1
    elif education == "Bachelor's":
        return 2
    elif education == "Master's":
        return 3
    elif education == "PhD":
        return 4
    elif education == "Other":
        return 0
    else:
        return "FAILED"

#assess employment
def assess_employment(employment_status):
    if employment_status == "Student":
        return "Unemployed"
    else:
        return employment_status

#assess sleep quality
def assess_sleep_quality(sleep_hours):
    if sleep_hours < 5:
        return "Poor"
    elif sleep_hours >= 5 and sleep_hours < 7:
        return "Fair"
    elif sleep_hours >= 7 and sleep_hours < 9:
        return "Good"
    elif sleep_hours >= 9: 
        return "Excellent"
    else:
        return "FAILED"

#assessing physical health: phys activity, substance use, chronic illness, work stress
#score out of 10 with different weightings on the categories
#sleep hours are 70%, other's are 10% each 
def assess_phys_health(activity_hours, substance_use, chronic_illness, medication_use):
    score = 3
    daily_hours = activity_hours/7

    #physical activity builds up the rest of the score out of 10 (70%)
    if daily_hours < 0.18:
        score += 2.3
    elif daily_hours >= 0.18 and daily_hours <= 0.38:
        score += 4.6
    elif daily_hours > 0.38:
        score += 7

    #other factors will decrease the initial 30% accordingly 
    if substance_use == "Occasional":
        score -= 0.5
    elif substance_use == "Frequent":
        score -= 1 
    
    if chronic_illness == 1:
        score -= 1

    if medication_use == "Occasional":
        score -= 0.5
    elif medication_use == "Frequent":
        score -= 1
    
    if score < 4:
        return "Unhealthy"
    if score >= 4 and score <= 6:
        return "Moderate"
    if score > 6:
        return "Healthy"

df["Employment_Status"] = df["Employment_Status"].apply(assess_employment)
df["Medication_Use"] = df["Medication_Use"].apply(transform_medication_use)
df["Substance_Use"] = df["Substance_Use"].apply(transform_substance_use)
df["Education_Level"] = df["Education_Level"].apply(transform_education)

"""df["Sleep_Quality"] = df["Sleep_Hours"].apply(assess_sleep_quality)
df["Phys_Health"] = df.apply(
    lambda row: assess_phys_health(
        row["Physical_Activity_Hrs"],
        row["Substance_Use"],
        row["Chronic_Illnesses"],
        row["Medication_Use"]
    ),
    axis=1
)
#df.drop(["Sleep_Hours", "Medication_Use", "Substance_Use", "Chronic_Illnesses"], axis=1, inplace=True)
#df.drop(["Medication_Use", "Substance_Use", "Chronic_Illnesses"], axis=1, inplace=True)"""

df = pd.get_dummies(df, columns=['Employment_Status', 'Gender'])
df = df.astype(int)

df.to_csv("../staging/processed_anxiety_depression_data.csv", index=False)