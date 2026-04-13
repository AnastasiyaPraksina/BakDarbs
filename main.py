from pathlib import Path
from eda_functions import load_data, run_full_eda

CLEANED_DATA_FILE = Path("data/cleaned_data.csv")


def main():
    df = load_data(CLEANED_DATA_FILE)
    run_full_eda(df, label_col="anomaly_label")


if __name__ == "__main__":
    main()