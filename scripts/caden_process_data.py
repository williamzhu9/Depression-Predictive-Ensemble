import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_PATH = PROJECT_ROOT / "raw" / "depression_anxiety_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "staging" / "depression_anxiety_formatted.csv"
STAGING_DIR = OUTPUT_PATH.parent
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def bool_to_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
        .astype("Int64")
    )


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "school_year",
        "age",
        "bmi",
        "phq_score",
        "gad_score",
        "epworth_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = [
        "depressiveness",
        "suicidal",
        "depression_diagnosis",
        "depression_treatment",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatment",
        "sleepiness",
    ]
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = bool_to_int(df[col])


    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower()

    return df


def one_hot_encode(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    dummies = pd.get_dummies(df[cols], prefix=cols, dtype="uint8")
    df_enc = pd.concat([df.drop(columns=cols), dummies], axis=1)
    return df_enc



def split_and_save_train_test(df: pd.DataFrame, target_col: str, output_dir: Path,):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.20,
        random_state = 42,
        stratify=y,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = output_dir / "depression_anxiety_train.csv"
    test_path = output_dir / "depression_anxiety_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def main():
    df = pd.read_csv(INPUT_PATH)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = standardize_types(df)
    
    one_hot_cols = ['gender', 'who_bmi', 'depression_severity', 'anxiety_severity']
    
    df = one_hot_encode(df, one_hot_cols)
    
    df.to_csv(OUTPUT_PATH, index=False) 
    
    split_and_save_train_test(
        df=df,
        target_col="depression_diagnosis",
        output_dir = STAGING_DIR
    )


if __name__ == "__main__":
    main()
