from train_models.scripts.depression_anxiety_processor import main as process_main
from train_models.models.depression_anxiety_rf_model import main as rf_main
from train_models.models.depression_anxiety_xg_model import main as xgb_main


def main(raw_filename: str = "depression_anxiety_data.csv") -> None:
    process_main(raw_filename)

    rf_main()
    xgb_main()


if __name__ == "__main__":
    main("depression_anxiety_data.csv")
