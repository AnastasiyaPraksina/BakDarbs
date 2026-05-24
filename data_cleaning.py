from pathlib import Path
import pandas as pd
from paths import (
    RAW_DATA,
    CLEANED_DATA
)

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    df = df.drop(columns=[
        "cell_id",
        "cytodiffusion_classification_confidence",
        "cytodiffusion_anomaly_score",
        "labeller_confidence_score"
    ], errors="ignore")

    if "patient_sex" in df.columns:
        df["patient_sex"] = df["patient_sex"].astype("category").cat.codes
    categorical_cols = [
        "patient_age_group",
        "dataset_source",
        "staining_protocol",
        "microscope_model"
    ]

    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

    if len(existing_categorical_cols) > 0:
        df = pd.get_dummies(
            df,
            columns=existing_categorical_cols,
            drop_first=True
        )

    # Convert bool to int
    bool_columns = df.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)

    return df

def save_cleaned_data(df: pd.DataFrame, output_file=CLEANED_DATA) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

def run_data_cleaning(file_path: str) -> None:
    df = load_data(file_path)

    original_shape = df.shape
    cleaned_df = clean_data(df)
    cleaned_shape = cleaned_df.shape

    save_cleaned_data(cleaned_df)