from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from paths import (
    CLEANED_DATA,
    TRANSFORMED_DATA,
    LABELS,
    TRAIN_DATA,
    VALIDATION_DATA,
    TEST_DATA,
    TRAIN_LABELS,
    VALIDATION_LABELS,
    TEST_LABELS
)

RANDOM_STATE = 42
TARGET_ANOMALY_RATE = 0.05  # 5% anomalies in the final experimental dataset


def load_data(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[
        "cell_diameter_um",
        "cell_area_px",
        "cytoplasm_ratio",
        "chromatin_density",
        "eccentricity",
        "cytodiffusion_classification_confidence",
        "cytodiffusion_anomaly_score",
        "labeller_confidence_score",
    ], errors="ignore")

    return df


def get_columns_to_scale(df: pd.DataFrame) -> list[str]:
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
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns_to_scale = get_columns_to_scale(X_train)

    print("\n=== Scaling info ===")
    print(f"Columns to scale ({len(columns_to_scale)}):")
    print(columns_to_scale)

    X_train_scaled = X_train.copy()
    X_validation_scaled = X_validation.copy()
    X_test_scaled = X_test.copy()

    if not columns_to_scale:
        print("No non-binary numeric columns need scaling.")
        return X_train_scaled, X_validation_scaled, X_test_scaled

    scaler = StandardScaler()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_validation_scaled[columns_to_scale] = scaler.transform(X_validation[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_validation_scaled, X_test_scaled


def build_realistic_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    anomaly_rate: float,
    random_state: int
) -> pd.DataFrame:
    full_data = pd.concat([features, labels], axis=1)

    normal_pool = full_data[full_data["anomaly_label"] == 0].copy()
    anomaly_pool = full_data[full_data["anomaly_label"] == 1].copy()

    n_normal = len(normal_pool)
    n_anomaly = len(anomaly_pool)

    print("\n=== Available class counts in original dataset ===")
    print(f"Normal observations:  {n_normal}")
    print(f"Anomalous observations: {n_anomaly}")

    # Maximum dataset size possible under requested anomaly rate
    max_total_by_normal = int(n_normal / (1 - anomaly_rate))
    max_total_by_anomaly = int(n_anomaly / anomaly_rate)
    max_total_size = min(max_total_by_normal, max_total_by_anomaly)

    anomaly_count = int(round(max_total_size * anomaly_rate))
    normal_count = max_total_size - anomaly_count

    print("\n=== Realistic dataset construction ===")
    print(f"Target anomaly rate: {anomaly_rate:.2%}")
    print(f"Maximum possible total size: {max_total_size}")
    print(f"Selected normal observations: {normal_count}")
    print(f"Selected anomalous observations: {anomaly_count}")

    sampled_normal = normal_pool.sample(n=normal_count, random_state=random_state)
    sampled_anomaly = anomaly_pool.sample(n=anomaly_count, random_state=random_state)

    realistic_df = pd.concat([sampled_normal, sampled_anomaly], axis=0)
    realistic_df = realistic_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return realistic_df


def print_split_statistics(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame
) -> None:
    print("\n=== Split shapes ===")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test:       {X_test.shape}")

    print("\n=== Anomaly proportions ===")
    print(f"Train anomaly rate:      {y_train['anomaly_label'].mean():.4f}")
    print(f"Validation anomaly rate: {y_validation['anomaly_label'].mean():.4f}")
    print(f"Test anomaly rate:       {y_test['anomaly_label'].mean():.4f}")


def split_dataset(file_path: str | Path = CLEANED_DATA) -> None:
  
    df = load_data(file_path)

    print("=== Input dataset shape ===")
    print(df.shape)

    labels = df[["anomaly_label", "cell_type", "disease_category"]].copy()

    features = df.drop(columns=[
        "anomaly_label",
        "cell_type",
        "disease_category"
    ], errors="ignore")

    features = preprocess_data(features)

    print("\n=== Feature dataset shape after preprocessing ===")
    print(features.shape)

    print("\n=== Columns after preprocessing ===")
    print(features.columns.tolist())

    # Save full transformed dataset and labels before realistic subsampling
    features.to_csv(TRANSFORMED_DATA, index=False, encoding="utf-8-sig")
    labels.to_csv(LABELS, index=False, encoding="utf-8-sig")

    # Build realistic rare-anomaly dataset
    realistic_df = build_realistic_dataset(
        features=features,
        labels=labels,
        anomaly_rate=TARGET_ANOMALY_RATE,
        random_state=RANDOM_STATE
    )

    print("\n=== Realistic dataset shape ===")
    print(realistic_df.shape)
    print(f"Realistic anomaly rate: {realistic_df['anomaly_label'].mean():.4f}")
    X = realistic_df.drop(columns=["anomaly_label", "cell_type", "disease_category"])
    y = realistic_df[["anomaly_label", "cell_type", "disease_category"]].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y["anomaly_label"]
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp["anomaly_label"]
    )

    X_train, X_validation, X_test = scale_datasets(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test
    )

    print_split_statistics(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test
    )

    X_train.to_csv(TRAIN_DATA, index=False, encoding="utf-8-sig")
    X_validation.to_csv(VALIDATION_DATA, index=False, encoding="utf-8-sig")
    X_test.to_csv(TEST_DATA, index=False, encoding="utf-8-sig")

    y_train.to_csv(TRAIN_LABELS, index=False, encoding="utf-8-sig")
    y_validation.to_csv(VALIDATION_LABELS, index=False, encoding="utf-8-sig")
    y_test.to_csv(TEST_LABELS, index=False, encoding="utf-8-sig")

    print("\n=== Files saved successfully ===")
    print(f"Transformed full data: {TRANSFORMED_DATA}")
    print(f"Full labels:           {LABELS}")
    print(f"Train data:            {TRAIN_DATA}")
    print(f"Validation data:       {VALIDATION_DATA}")
    print(f"Test data:             {TEST_DATA}")
    print(f"Train labels:          {TRAIN_LABELS}")
    print(f"Validation labels:     {VALIDATION_LABELS}")
    print(f"Test labels:           {TEST_LABELS}")

if __name__ == "__main__":
    split_dataset()