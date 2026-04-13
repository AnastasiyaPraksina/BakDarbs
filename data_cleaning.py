from pathlib import Path
import pandas as pd
from paths import raw_data


DATA_DIR = Path("data")
CLEANED_DATA_FILE = DATA_DIR / "cleaned_data.csv"


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names (remove spaces/BOM issues)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    # Drop non-informative and leakage-prone columns
    df = df.drop(columns=[
        "dataset_source",
        "microscope_model",
        "staining_protocol",
        "magnification_x",
        "image_resolution_px",
        "cell_id",
        "cytodiffusion_classification_confidence",
        "cytodiffusion_anomaly_score",
        "labeller_confidence_score"
    ], errors="ignore")

    return df


def save_cleaned_data(df: pd.DataFrame, output_file: Path = CLEANED_DATA_FILE) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")


def run_data_cleaning(file_path: str) -> None:
    df = load_data(file_path)

    original_shape = df.shape

    cleaned_df = clean_data(df)
    cleaned_shape = cleaned_df.shape

    save_cleaned_data(cleaned_df)

    print("=== Data cleaning summary ===")
    print(f"Original: rows={original_shape[0]}, columns={original_shape[1]}")
    print(f"Cleaned:  rows={cleaned_shape[0]}, columns={cleaned_shape[1]}")


if __name__ == "__main__":
    run_data_cleaning(raw_data)