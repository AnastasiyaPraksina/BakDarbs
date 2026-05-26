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
TARGET_ANOMALY_RATE = 0.05

# This script prepares the dataset before anomaly detection experiments.
# First, unnecessary or potentially leaking features are removed from  the dataset. After that, the data is separated into features and labels.
# A more realistic dataset is then created with a fixed anomaly ratio so the experiments better represent real healthcare scenarios where anomalies are rare.
# The dataset is split into training, validation and test sets using stratified splitting to keep anomaly proportions similar in each part.
# Numerical features are scaled with StandardScaler, while binary attributes are left unchanged.

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    cols_to_drop = [
        "cell_diameter_um",
        "cell_area_px",
        "cytoplasm_ratio",
        "chromatin_density",
        "eccentricity",
        "cytodiffusion_classification_confidence",
        "cytodiffusion_anomaly_score",
        "labeller_confidence_score"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def get_columns_to_scale(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns

    cols = []

    for col in numeric_cols:
        values = set(df[col].dropna().unique())

        if values.issubset({0, 1}):
            continue

        cols.append(col)

    return cols


def scale_datasets(X_train, X_validation, X_test):
    cols = get_columns_to_scale(X_train)

    X_train_scaled = X_train.copy()
    X_validation_scaled = X_validation.copy()
    X_test_scaled = X_test.copy()

    if len(cols) == 0:
        return X_train_scaled, X_validation_scaled, X_test_scaled

    scaler = StandardScaler()

    X_train_scaled[cols] = scaler.fit_transform(X_train[cols])

    X_validation_scaled[cols] = scaler.transform(
        X_validation[cols]
    )

    X_test_scaled[cols] = scaler.transform(
        X_test[cols]
    )

    return X_train_scaled, X_validation_scaled, X_test_scaled


def build_realistic_dataset(features, labels, anomaly_rate, random_state):
    df = pd.concat([features, labels], axis=1)

    normal = df[df["anomaly_label"] == 0]
    anomaly = df[df["anomaly_label"] == 1]

    normal_count = len(normal)
    anomaly_count = len(anomaly)

    max_total_1 = int(normal_count / (1 - anomaly_rate))
    max_total_2 = int(anomaly_count / anomaly_rate)

    max_total = min(max_total_1, max_total_2)

    selected_anomaly = int(max_total * anomaly_rate)
    selected_normal = max_total - selected_anomaly

    normal_sample = normal.sample(
        n=selected_normal,
        random_state=random_state
    )

    anomaly_sample = anomaly.sample(
        n=selected_anomaly,
        random_state=random_state
    )

    result = pd.concat([
        normal_sample,
        anomaly_sample
    ])

    result = result.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    return result


def split_dataset(file_path=CLEANED_DATA):
    df = load_data(file_path)

    labels = df[
        ["anomaly_label", "cell_type", "disease_category"]
    ].copy()

    features = df.drop(
        columns=[
            "anomaly_label",
            "cell_type",
            "disease_category"
        ],
        errors="ignore"
    )

    features = preprocess_data(features)

    features.to_csv(
        TRANSFORMED_DATA,
        index=False,
        encoding="utf-8-sig"
    )

    labels.to_csv(
        LABELS,
        index=False,
        encoding="utf-8-sig"
    )

    realistic_df = build_realistic_dataset(
        features,
        labels,
        TARGET_ANOMALY_RATE,
        RANDOM_STATE
    )

    X = realistic_df.drop(
        columns=[
            "anomaly_label",
            "cell_type",
            "disease_category"
        ]
    )

    y = realistic_df[
        ["anomaly_label", "cell_type", "disease_category"]
    ].copy()

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
        X_train,
        X_validation,
        X_test
    )

    X_train.to_csv(
        TRAIN_DATA,
        index=False,
        encoding="utf-8-sig"
    )

    X_validation.to_csv(
        VALIDATION_DATA,
        index=False,
        encoding="utf-8-sig"
    )

    X_test.to_csv(
        TEST_DATA,
        index=False,
        encoding="utf-8-sig"
    )

    y_train.to_csv(
        TRAIN_LABELS,
        index=False,
        encoding="utf-8-sig"
    )

    y_validation.to_csv(
        VALIDATION_LABELS,
        index=False,
        encoding="utf-8-sig"
    )

    y_test.to_csv(
        TEST_LABELS,
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    split_dataset()