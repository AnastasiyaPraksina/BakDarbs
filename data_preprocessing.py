from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")

CLEANED_DATA_FILE = DATA_DIR / "cleaned_data.csv"
TRANSFORMED_DATA_FILE = DATA_DIR / "transformed_data.csv"
LABELS_FILE = DATA_DIR / "labels.csv"

TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
TEST_DATA_FILE = DATA_DIR / "test_data.csv"

TRAIN_LABELS_FILE = DATA_DIR / "train_labels.csv"
TEST_LABELS_FILE = DATA_DIR / "test_labels.csv"


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categorical features using one-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            "patient_age_group",
            "patient_sex"
        ],
        drop_first=True
    )

    # Convert boolean dummy columns to integers
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    return df


def get_columns_to_scale(df: pd.DataFrame) -> list:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    columns_to_scale = []

    for col in numeric_columns:
        unique_values = set(df[col].dropna().unique())

        if unique_values.issubset({0, 1}):
            continue

        columns_to_scale.append(col)

    return columns_to_scale


def scale_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    columns_to_scale = get_columns_to_scale(X_train)

    print("\n=== Scaling info ===")
    print(f"Columns to scale ({len(columns_to_scale)}): {columns_to_scale}")

    if not columns_to_scale:
        print("No columns need scaling.")
        return X_train, X_test

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled


def split_dataset(file_path: str = CLEANED_DATA_FILE) -> None:
    df = load_data(file_path)

    print("=== Cleaned dataset shape ===")
    print(df.shape)

    labels = df[["anomaly_label", "cell_type", "disease_category"]].copy()

    transformed_data = df.drop(columns=[
        "anomaly_label",
        "cell_type",
        "disease_category"
    ], errors="ignore")

    transformed_data = preprocess_data(transformed_data)

    X_train, X_test, y_train, y_test = train_test_split(
        transformed_data,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels["anomaly_label"]
    )

    X_train, X_test = scale_datasets(
        X_train=X_train,
        X_test=X_test,
    )

    # Save datasets
    transformed_data.to_csv(TRANSFORMED_DATA_FILE, index=False, encoding="utf-8-sig")
    labels.to_csv(LABELS_FILE, index=False, encoding="utf-8-sig")

    X_train.to_csv(TRAIN_DATA_FILE, index=False, encoding="utf-8-sig")
    X_test.to_csv(TEST_DATA_FILE, index=False, encoding="utf-8-sig")

    y_train.to_csv(TRAIN_LABELS_FILE, index=False, encoding="utf-8-sig")
    y_test.to_csv(TEST_LABELS_FILE, index=False, encoding="utf-8-sig")

    print("\n=== Dataset split info ===")
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")


if __name__ == "__main__":
    split_dataset()