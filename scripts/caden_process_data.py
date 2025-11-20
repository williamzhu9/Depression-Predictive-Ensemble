import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_PATH = PROJECT_ROOT / "raw" / "depression_anxiety_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "staging" / "depression_anxiety_standardized.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)



def standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:

    if "school_year" in df.columns:
        df["education_level"] = "bachelors degree"   # your requirement
    

    df["employment_status_standard"] = "unemployed"


    df = df.drop(columns=["school_year", "education_current"], errors="ignore")

    return df



def standardize_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sleep quality must be derived directly from Epworth, with a 4-category scale.
    Categories aligned to Copper's clinical interpretation but mapped into the
    user's 4 discrete bins:
        - <5  → poor
        - 5-6 → fair
        - 7-8 → good
        - 9+  → excellent
    """

    def classify_sleep(row):
        epw = row.get("epworth_score")
        sleepy = row.get("sleepiness")

        try:
            epw = float(epw)
        except (TypeError, ValueError):
            epw = None

 
        if epw is not None:
            if epw <= 3:
                return "excellent"      
            elif 4 <= epw <= 6:
                return "good"           
            elif 7 <= epw <= 8:
                return "fair"           
            elif epw >= 9:
                return "poor"           


        sleepy_str = str(sleepy).strip().lower()
        if sleepy_str == "true":
            return "fair"
        elif sleepy_str == "false":
            return "good"

        return "unknown"

    df["sleep_quality_cat"] = df.apply(classify_sleep, axis=1)
    return df

def standardize_bmi(df: pd.DataFrame) -> pd.DataFrame:
    if "who_bmi" in df.columns:
        health_score_map = {
            "Normal": 3,
            "Underweight": 1,
            "Overweight": 2,
            "Class I Obesity": 1,
            "Class II Obesity": 0,
            "Class III Obesity": 0,
            "Not Availble": None,
            "Not Available": None,
        }
        df["physical_health_score"] = df["who_bmi"].map(health_score_map)
    return df


def _parse_bool_like(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def standardize_depression(df: pd.DataFrame) -> pd.DataFrame:
    if "phq_score" in df.columns:


        bins = [-1, 4, 9, 14, 19, 27]
        labels = [0, 1, 2, 3, 4]

        df["depression_severity_score"] = pd.cut(
            df["phq_score"], bins=bins, labels=labels
        ).astype("Int64")

        df["depression_any_symptoms"] = (df["phq_score"] >= 5).astype("Int64")

        if "depression_diagnosis" in df.columns:
            df["depression_diagnosed"] = _parse_bool_like(df["depression_diagnosis"]).astype("Int64")

        if "depression_treatment" in df.columns:
            df["depression_treated"] = _parse_bool_like(df["depression_treatment"]).astype("Int64")

    return df


def standardize_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    if "gad_score" in df.columns:

        bins = [-1, 4, 9, 14, 21]
        labels = [0, 1, 2, 3]

        df["anxiety_severity_score"] = pd.cut(
            df["gad_score"], bins=bins, labels=labels
        ).astype("Int64")

        df["anxiety_any_symptoms"] = (df["gad_score"] >= 5).astype("Int64")

        if "anxiety_diagnosis" in df.columns:
            df["anxiety_diagnosed"] = _parse_bool_like(df["anxiety_diagnosis"]).astype("Int64")

        if "anxiety_treatment" in df.columns:
            df["anxiety_treated"] = _parse_bool_like(df["anxiety_treatment"]).astype("Int64")

    return df


def main():
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()

    df = standardize_demographics(df)
    df = standardize_sleep(df)
    df = standardize_bmi(df)
    df = standardize_depression(df)
    df = standardize_anxiety(df)

    
    cols_to_drop = [
        "id",
        "bmi",
        "who_bmi",
        "depression_severity",
        "depressiveness",
        "suicidal",
        "depression_diagnosis",
        "depression_treatment",
        "anxiety_severity",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatment",
        "sleepiness",
        "epworth_score",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved standardized file to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
