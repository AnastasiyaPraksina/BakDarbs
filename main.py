from paths import raw_data
from eda_functions import load_data, run_full_eda


def main():
    df = load_data(raw_data)
    run_full_eda(df, label_col="anomaly_label")


if __name__ == "__main__":
    main()