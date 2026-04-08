import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# PAMATA IESTATĪJUMI
# =========================

sns.set_theme(style="whitegrid")

PLOTS_DIR = Path("plots")

PLOT_FOLDERS = {
    "label_distribution": PLOTS_DIR / "label_distribution",
    "feature_histograms": PLOTS_DIR / "feature_histograms",
    "feature_boxplots": PLOTS_DIR / "feature_boxplots",
    "boxplots_by_label": PLOTS_DIR / "boxplots_by_label",
    "histograms_by_label": PLOTS_DIR / "histograms_by_label",
    "scatterplots": PLOTS_DIR / "scatterplots",
    "correlation": PLOTS_DIR / "correlation",
}


def create_plot_folders() -> None:
    for folder in PLOT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]+', "_", str(name))


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def show_dataset_info(df: pd.DataFrame) -> None:
    print("=== Datu kopas izmērs ===")
    print(df.shape)

    print("\n=== Kolonnas ===")
    print(df.columns.tolist())

    print("\n=== Datu tipi ===")
    print(df.dtypes)

    print("\n=== Trūkstošo vērtību skaits ===")
    print(df.isnull().sum())


def save_dataset_statistics(df: pd.DataFrame, output_file: str = "dataset_statistics.csv") -> None:
    """Saglabā kopējo datu kopas statistiku CSV failā projekta saknē."""
    stats_rows = []

    for column in df.columns:
        series = df[column]
        non_null_series = series.dropna()

        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "non_null_count": int(non_null_series.shape[0]),
            "unique_count": int(non_null_series.nunique()),
            "minimum": np.nan,
            "maximum": np.nan,
            "mode": np.nan,
            "median": np.nan,
            "percentile_25": np.nan,
            "percentile_50": np.nan,
            "percentile_75": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "geometric_mean": np.nan,
            "harmonic_mean": np.nan,
            "missing_count": int(series.isnull().sum()),
            "missing_percentage": round(series.isnull().mean() * 100, 4),
        }

        if not non_null_series.empty:
            mode_values = non_null_series.mode()
            if not mode_values.empty:
                row["mode"] = mode_values.iloc[0]

        if pd.api.types.is_numeric_dtype(series):
            row["minimum"] = non_null_series.min() if not non_null_series.empty else np.nan
            row["maximum"] = non_null_series.max() if not non_null_series.empty else np.nan
            row["median"] = non_null_series.median() if not non_null_series.empty else np.nan
            row["percentile_25"] = non_null_series.quantile(0.25) if not non_null_series.empty else np.nan
            row["percentile_50"] = non_null_series.quantile(0.50) if not non_null_series.empty else np.nan
            row["percentile_75"] = non_null_series.quantile(0.75) if not non_null_series.empty else np.nan
            row["mean"] = non_null_series.mean() if not non_null_series.empty else np.nan
            row["std"] = non_null_series.std() if not non_null_series.empty else np.nan

            positive_values = non_null_series[non_null_series > 0]

            if not positive_values.empty:
                row["geometric_mean"] = np.exp(np.log(positive_values).mean())
                row["harmonic_mean"] = len(positive_values) / (1 / positive_values).sum()

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nStatistikas fails saglabāts: {output_file}")


def get_numeric_features(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
    if exclude_columns is None:
        exclude_columns = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    return numeric_cols


def get_non_constant_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Atgriež tikai tos atribūtus, kuriem ir vairāk nekā viena unikāla vērtība."""
    return [col for col in features if df[col].nunique(dropna=True) > 1]


# =========================
# 01. LABEL DISTRIBUTION
# =========================

def plot_label_distribution(df: pd.DataFrame, label_col: str = "anomaly_label") -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=label_col)
    plt.title("Klašu sadalījums")
    plt.xlabel("Klase")
    plt.ylabel("Biežums")
    plt.tight_layout()

    save_path = PLOT_FOLDERS["label_distribution"] / "label_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 02. FEATURE HISTOGRAMS
# =========================

def plot_feature_histograms(df: pd.DataFrame, features: List[str]) -> None:
    for feature in features:
        feature_data = df[feature].dropna()

        if feature_data.empty:
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(feature_data, kde=True)
        plt.title(f"Atribūta {feature} sadalījums")
        plt.xlabel(feature)
        plt.ylabel("Biežums")
        plt.tight_layout()

        save_path = PLOT_FOLDERS["feature_histograms"] / f"{sanitize_filename(feature)}_histogram.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# 03. FEATURE BOXPLOTS
# =========================

def plot_feature_boxplots(df: pd.DataFrame, features: List[str]) -> None:
    for feature in features:
        feature_data = df[feature].dropna()

        if feature_data.empty:
            continue

        plt.figure(figsize=(7, 5))
        sns.boxplot(y=feature_data)
        plt.title(f"Atribūta {feature} boxplot diagramma")
        plt.ylabel(feature)
        plt.tight_layout()

        save_path = PLOT_FOLDERS["feature_boxplots"] / f"{sanitize_filename(feature)}_boxplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# 04. BOXPLOTS BY LABEL
# =========================

def plot_boxplots_by_label(df: pd.DataFrame, features: List[str], label_col: str = "anomaly_label") -> None:
    for feature in features:
        plot_df = df[[feature, label_col]].dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=plot_df, x=label_col, y=feature)
        plt.title(f"{feature} sadalījums pa klasēm")
        plt.xlabel("Klase")
        plt.ylabel(feature)
        plt.tight_layout()

        save_path = PLOT_FOLDERS["boxplots_by_label"] / f"{sanitize_filename(feature)}_boxplot_by_label.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# 05. HISTOGRAMS BY LABEL
# =========================

def plot_histograms_by_label(df: pd.DataFrame, features: List[str], label_col: str = "anomaly_label") -> None:
    for feature in features:
        plot_df = df[[feature, label_col]].dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(data=plot_df, x=feature, hue=label_col, kde=True, element="step")
        plt.title(f"{feature} sadalījums pa klasēm")
        plt.xlabel(feature)
        plt.ylabel("Biežums")
        plt.tight_layout()

        save_path = PLOT_FOLDERS["histograms_by_label"] / f"{sanitize_filename(feature)}_histogram_by_label.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# 07. CORRELATION
# =========================

def plot_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    corr_matrix = df[features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True)
    plt.title("Atribūtu korelācijas matrica")
    plt.tight_layout()

    save_path = PLOT_FOLDERS["correlation"] / "correlation_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    corr_csv_path = PLOT_FOLDERS["correlation"] / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv_path, encoding="utf-8-sig")

    return corr_matrix


# =========================
# SCATTER PAIR SELECTION
# =========================

def get_correlated_feature_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
    max_pairs: int = 15
) -> List[Tuple[str, str, float]]:
    pairs = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            corr_value = corr_matrix.loc[feature_1, feature_2]

            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                pairs.append((feature_1, feature_2, corr_value))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:max_pairs]


# =========================
# 06. SCATTERPLOTS
# =========================

def plot_scatterplots(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str, float]],
    label_col: str = "anomaly_label"
) -> None:
    for feature_1, feature_2, corr_value in feature_pairs:
        plot_df = df[[feature_1, feature_2, label_col]].dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x=feature_1,
            y=feature_2,
            hue=label_col,
            alpha=0.7
        )
        plt.title(f"{feature_1} pret {feature_2} (r = {corr_value:.2f})")
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.tight_layout()

        filename = f"{sanitize_filename(feature_1)}_vs_{sanitize_filename(feature_2)}_scatter.png"
        save_path = PLOT_FOLDERS["scatterplots"] / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================
# PILNS EDA PROCESS
# =========================

def run_full_eda(df: pd.DataFrame, label_col: str = "anomaly_label") -> None:
    create_plot_folders()
    show_dataset_info(df)
    save_dataset_statistics(df)

    if label_col not in df.columns:
        print(f"Kolonna '{label_col}' netika atrasta datu kopā.")
        return

    numeric_features = get_numeric_features(df, exclude_columns=[label_col])

    if not numeric_features:
        print("Nav atrasti skaitliskie atribūti analīzei.")
        return

    numeric_features_non_constant = get_non_constant_features(df, numeric_features)

    print("\nTiek veidoti grafiki...")

    plot_label_distribution(df, label_col=label_col)
    plot_feature_histograms(df, numeric_features)
    plot_feature_boxplots(df, numeric_features)
    plot_boxplots_by_label(df, numeric_features, label_col=label_col)
    plot_histograms_by_label(df, numeric_features, label_col=label_col)

    if len(numeric_features_non_constant) >= 2:
        corr_matrix = plot_correlation_heatmap(df, numeric_features_non_constant)
        scatter_pairs = get_correlated_feature_pairs(corr_matrix)

        if scatter_pairs:
            plot_scatterplots(df, scatter_pairs, label_col=label_col)
            print(f"Scatter diagrammām atlasīti {len(scatter_pairs)} atribūtu pāri.")
        else:
            print("Nav atrasti atribūtu pāri ar pietiekamu korelāciju scatter diagrammām.")
    else:
        print("Korelācijas matricai un scatter diagrammām nepietiek nekonstantu skaitlisko atribūtu.")

    print("\nEDA pabeigts. Visi grafiki saglabāti mapē 'plots/'.")