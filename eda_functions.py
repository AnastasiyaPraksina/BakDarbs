import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from paths import (
    CLEANED_DATA,
    PLOTS_EDA_DIR
)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

sns.set_theme(style="whitegrid")

PLOTS_DIR = PLOTS_EDA_DIR

PLOT_FOLDERS = {
    "label_distribution": PLOTS_DIR / "label_distribution",
    "feature_histograms": PLOTS_DIR / "feature_histograms",
    "feature_boxplots": PLOTS_DIR / "feature_boxplots",
    "boxplots_by_label": PLOTS_DIR / "boxplots_by_label",
    "histograms_by_label": PLOTS_DIR / "histograms_by_label",
    "scatterplots": PLOTS_DIR / "scatterplots",
    "correlation": PLOTS_DIR / "correlation",
    "categorical_countplots": PLOTS_DIR / "categorical_countplots",
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


def check_duplicates(df: pd.DataFrame, name: str = "Dataset") -> None:
    duplicates = df.duplicated().sum()

    print(f"\n=== Duplicate check for {name} ===")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {duplicates}")

    if duplicates > 0:
        print("Warning: Dataset contains full duplicate rows!")
    else:
        print("No full duplicates found.")


def get_numeric_features(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None
) -> List[str]:
    if exclude_columns is None:
        exclude_columns = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    return numeric_cols


def get_binary_features(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None
) -> List[str]:
    if exclude_columns is None:
        exclude_columns = []

    binary_cols = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        unique_vals = df[col].dropna().unique()

        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)

    return binary_cols


def get_non_constant_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    return [col for col in features if df[col].nunique(dropna=True) > 1]


def compute_pearson_kurtosis(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty or clean.nunique() <= 1:
        return np.nan

    std = clean.std(ddof=0)
    if std == 0 or pd.isna(std):
        return np.nan

    z = (clean - clean.mean()) / std
    return float(np.mean(z ** 4))


def analyze_numeric_features_kurtosis(
    df: pd.DataFrame,
    numeric_features: List[str]
) -> Dict[str, Dict[str, Any]]:
    results = {}

    for feature in numeric_features:
        series = pd.to_numeric(df[feature], errors="coerce")
        pearson_kurtosis = compute_pearson_kurtosis(series)

        if pd.notna(pearson_kurtosis):
            excess_kurtosis = pearson_kurtosis - 3.0
        else:
            excess_kurtosis = np.nan

        if pd.isna(pearson_kurtosis):
            interpretation = "not_applicable"
        elif pearson_kurtosis > 3:
            interpretation = "high_tail_weight"
        elif np.isclose(pearson_kurtosis, 3, atol=0.3):
            interpretation = "approximately_normal_like"
        else:
            interpretation = "light_tails"

        results[feature] = {
            "feature_analysis_type": "numeric_kurtosis",
            "pearson_kurtosis": pearson_kurtosis,
            "excess_kurtosis": excess_kurtosis,
            "kurtosis_interpretation": interpretation,
        }

    return results


def analyze_binary_features_statistical(
    df: pd.DataFrame,
    binary_features: List[str],
    exclude_columns: Optional[List[str]] = None,
    low_variance_threshold: float = 0.01,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Statistical analysis for binary features:
    1) Variance of binary feature p*(1-p)
    2) Mean mutual information with other features
    """
    if exclude_columns is None:
        exclude_columns = []

    results = {}

    for feature in binary_features:
        try:
            series = df[feature].dropna()

            if series.empty or series.nunique() < 2:
                results[feature] = {
                    "feature_analysis_type": "binary_statistical",
                    "binary_positive_rate": np.nan,
                    "binary_variance": np.nan,
                    "binary_variance_flag": "not_applicable",
                    "binary_mean_mutual_information": np.nan,
                    "binary_mi_interpretation": "not_applicable",
                }
                continue

            p = float(series.mean())
            variance = float(p * (1 - p))

            if variance < low_variance_threshold:
                variance_flag = "low_variance"
            else:
                variance_flag = "acceptable_variance"

            predictor_cols = [
                col for col in df.columns
                if col != feature and col not in exclude_columns
            ]

            X = df[predictor_cols].copy()
            y = df[feature].copy()

            valid_mask = y.notna()
            X = X.loc[valid_mask].copy()
            y = y.loc[valid_mask].astype(int).copy()

            X = pd.get_dummies(X, drop_first=True)

            bool_cols = X.select_dtypes(include=["bool"]).columns
            if len(bool_cols) > 0:
                X[bool_cols] = X[bool_cols].astype(int)

            X = X.select_dtypes(include=[np.number])

            if X.empty or y.nunique() < 2:
                mean_mi = np.nan
                mi_interpretation = "not_enough_predictors"
            else:
                medians = X.median(numeric_only=True)
                X = X.fillna(medians)

                discrete_mask = [
                    pd.api.types.is_integer_dtype(X[col]) and X[col].nunique(dropna=True) <= 10
                    for col in X.columns
                ]

                mi_scores = mutual_info_classif(
                    X=X,
                    y=y,
                    discrete_features=discrete_mask,
                    random_state=random_state
                )

                if len(mi_scores) == 0:
                    mean_mi = np.nan
                    mi_interpretation = "not_evaluable"
                else:
                    mean_mi = float(np.mean(mi_scores))

                    if mean_mi >= 0.10:
                        mi_interpretation = "high_shared_information"
                    elif mean_mi >= 0.03:
                        mi_interpretation = "moderate_shared_information"
                    else:
                        mi_interpretation = "low_shared_information"

            results[feature] = {
                "feature_analysis_type": "binary_statistical",
                "binary_positive_rate": p,
                "binary_variance": variance,
                "binary_variance_flag": variance_flag,
                "binary_mean_mutual_information": mean_mi,
                "binary_mi_interpretation": mi_interpretation,
            }

        except Exception as e:
            results[feature] = {
                "feature_analysis_type": "binary_statistical",
                "binary_positive_rate": np.nan,
                "binary_variance": np.nan,
                "binary_variance_flag": f"error: {str(e)}",
                "binary_mean_mutual_information": np.nan,
                "binary_mi_interpretation": f"error: {str(e)}",
            }

    return results


def save_dataset_statistics(
    df: pd.DataFrame,
    numeric_analysis: Optional[Dict[str, Dict[str, Any]]] = None,
    binary_analysis: Optional[Dict[str, Dict[str, Any]]] = None,
    output_file: Path = PLOTS_DIR / "dataset_statistics.csv"
) -> None:
    stats_rows = []

    numeric_analysis = numeric_analysis or {}
    binary_analysis = binary_analysis or {}

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

            "feature_analysis_type": np.nan,
            "pearson_kurtosis": np.nan,
            "excess_kurtosis": np.nan,
            "kurtosis_interpretation": np.nan,

            "binary_positive_rate": np.nan,
            "binary_variance": np.nan,
            "binary_variance_flag": np.nan,
            "binary_mean_mutual_information": np.nan,
            "binary_mi_interpretation": np.nan,
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

            positive_values = pd.to_numeric(non_null_series, errors="coerce")
            positive_values = positive_values[positive_values > 0]

            if not positive_values.empty:
                row["geometric_mean"] = np.exp(np.log(positive_values).mean())
                row["harmonic_mean"] = len(positive_values) / (1 / positive_values).sum()

        if column in numeric_analysis:
            row.update(numeric_analysis[column])

        if column in binary_analysis:
            row.update(binary_analysis[column])

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nStatistikas fails saglabāts: {output_file}")


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


def plot_boxplots_by_label(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "anomaly_label"
) -> None:
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


def plot_histograms_by_label(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "anomaly_label"
) -> None:
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


def plot_binary_countplots(
    df: pd.DataFrame,
    features: List[str],
    label_col: str = "anomaly_label"
) -> None:
    for feature in features:
        plot_df = df[[feature, label_col]].dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(7, 5))
        sns.countplot(data=plot_df, x=feature, hue=label_col)
        plt.title(f"{feature} sadalījums pa klasēm")
        plt.xlabel(feature)
        plt.ylabel("Biežums")
        plt.tight_layout()

        save_path = PLOT_FOLDERS["categorical_countplots"] / f"{sanitize_filename(feature)}_countplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_scatterplots(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str, float]],
    label_col: str = "anomaly_label"
) -> None:
    for feature_1, feature_2, corr_value in feature_pairs:
        if not pd.api.types.is_numeric_dtype(df[feature_1]) or not pd.api.types.is_numeric_dtype(df[feature_2]):
            continue

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


def plot_correlation_heatmap(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    if len(numeric_features) < 2:
        return pd.DataFrame()

    numeric_df = df[numeric_features].copy()
    corr_matrix = numeric_df.corr(numeric_only=True)

    if corr_matrix.empty:
        return pd.DataFrame()

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


def get_correlated_feature_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
    max_pairs: int = 20
) -> List[Tuple[str, str, float]]:
    if corr_matrix.empty:
        return []

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

def build_numeric_feature_selection(
    numeric_features: List[str],
    numeric_analysis: Dict[str, Dict[str, Any]],
    corr_matrix: pd.DataFrame,
    kurtosis_threshold: float = 3.0,
    correlation_threshold: float = 0.7,
    max_group_size: int = 3
) -> pd.DataFrame:
    rows = []

    if not numeric_features:
        return pd.DataFrame()

    def get_kurtosis(feature: str) -> float:
        return numeric_analysis.get(feature, {}).get("pearson_kurtosis", np.nan)

    def get_kurtosis_level(k: float) -> str:
        if pd.isna(k):
            return "unknown"
        return "high" if k > kurtosis_threshold else "low"

    if corr_matrix.empty:
        for feature in numeric_features:
            k = get_kurtosis(feature)
            rows.append({
                "feature": feature,
                "pearson_kurtosis": k,
                "kurtosis_level": get_kurtosis_level(k),
                "max_abs_correlation": np.nan,
                "most_correlated_feature": np.nan,
                "correlation_level": "low",
                "correlated_group_id": np.nan,
                "group_size": 1,
                "selected_within_group": "yes",
                "decision": "keep",
                "justification": "No strong correlation matrix was available; the feature is retained."
            })
        return pd.DataFrame(rows)

    # Step 1. Build list of strong pairs
    strong_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            f1 = numeric_features[i]
            f2 = numeric_features[j]

            if f1 not in corr_matrix.index or f2 not in corr_matrix.columns:
                continue

            corr_value = corr_matrix.loc[f1, f2]
            if pd.notna(corr_value) and abs(corr_value) >= correlation_threshold:
                strong_pairs.append((f1, f2, abs(float(corr_value))))

    strong_pairs.sort(key=lambda x: x[2], reverse=True)

    groups: List[List[str]] = []

    for f1, f2, _ in strong_pairs:
        placed = False

        for group in groups:
            group_set = set(group)
            if len(group) >= max_group_size:
                continue

            if f1 in group_set and f2 in group_set:
                placed = True
                break
            if f1 in group_set and f2 not in group_set:
                can_add = True
                for member in group:
                    if abs(corr_matrix.loc[f2, member]) < correlation_threshold:
                        can_add = False
                        break
                if can_add and len(group) < max_group_size:
                    group.append(f2)
                placed = True
                break
            if f2 in group_set and f1 not in group_set:
                can_add = True
                for member in group:
                    if abs(corr_matrix.loc[f1, member]) < correlation_threshold:
                        can_add = False
                        break
                if can_add and len(group) < max_group_size:
                    group.append(f1)
                placed = True
                break
        if not placed:
            groups.append([f1, f2])

    # Step 3. Clean overlaps greedily
    cleaned_groups: List[List[str]] = []
    used_features = set()

    def group_score(group: List[str]) -> Tuple[int, float]:
        kurt_sum = 0.0
        for f in group:
            k = get_kurtosis(f)
            if pd.notna(k):
                kurt_sum += k
        return (len(group), kurt_sum)

    groups = [sorted(list(set(g))) for g in groups]
    groups.sort(key=group_score, reverse=True)

    for group in groups:
        filtered = [f for f in group if f not in used_features]
        if len(filtered) >= 2:
            cleaned_groups.append(filtered)
            used_features.update(filtered)

    groups = cleaned_groups

    grouped_features = set(f for group in groups for f in group)

    feature_to_group_id = {}
    group_survivor = {}

    for group_id, group in enumerate(groups, start=1):
        scored = []
        for f in group:
            k = get_kurtosis(f)
            if pd.isna(k):
                k = -np.inf
            scored.append((f, k))

        scored.sort(key=lambda x: (-x[1], x[0]))
        survivor = scored[0][0]

        group_survivor[group_id] = survivor

        for f in group:
            feature_to_group_id[f] = group_id

    # Step 5. Build final rows
    for feature in numeric_features:
        k = get_kurtosis(feature)
        kurtosis_level = get_kurtosis_level(k)

        max_abs_corr = np.nan
        most_correlated_feature = np.nan
        correlation_level = "low"

        if feature in corr_matrix.columns:
            correlations = corr_matrix[feature].drop(labels=[feature], errors="ignore").dropna()
            if not correlations.empty:
                abs_corr = correlations.abs()
                max_abs_corr = float(abs_corr.max())
                most_correlated_feature = abs_corr.idxmax()
                if max_abs_corr >= correlation_threshold:
                    correlation_level = "high"

        if feature in feature_to_group_id:
            group_id = feature_to_group_id[feature]
            current_group = groups[group_id - 1]
            group_size = len(current_group)
            survivor = group_survivor[group_id]

            if feature == survivor:
                selected_within_group = "yes"
                decision = "keep_one_from_group"
                justification = (
                    f"Feature belongs to strict correlation group {group_id} (size={group_size}) "
                    f"and is retained because it has the highest kurtosis within this group."
                )
            else:
                selected_within_group = "no"
                decision = "drop"
                justification = (
                    f"Feature belongs to strict correlation group {group_id} (size={group_size}) "
                    f"and is dropped because another feature in the same group has higher kurtosis."
                )
        else:
            group_id = np.nan
            group_size = 1
            selected_within_group = "yes"
            decision = "keep"

            if correlation_level == "high":
                justification = (
                    "Feature has strong pairwise correlation with another feature, "
                    "but it was not included into a strict limited-size correlation group; retained for context."
                )
            elif kurtosis_level == "high":
                justification = (
                    "Feature is not part of a strict strong-correlation group and has high kurtosis, "
                    "therefore it is retained as a unique anomaly-related feature."
                )
            else:
                justification = (
                    "Feature is not part of a strict strong-correlation group and is retained "
                    "as a non-redundant contextual feature."
                )

        rows.append({
            "feature": feature,
            "pearson_kurtosis": k,
            "kurtosis_level": kurtosis_level,
            "max_abs_correlation": max_abs_corr,
            "most_correlated_feature": most_correlated_feature,
            "correlation_level": correlation_level,
            "correlated_group_id": group_id,
            "group_size": group_size,
            "selected_within_group": selected_within_group,
            "decision": decision,
            "justification": justification,
        })

    result_df = pd.DataFrame(rows)

    decision_order = {
        "drop": 0,
        "keep_one_from_group": 1,
        "keep": 2,
    }

    result_df["decision_priority"] = result_df["decision"].map(decision_order).fillna(99)
    result_df = result_df.sort_values(
        by=["decision_priority", "correlated_group_id", "pearson_kurtosis"],
        ascending=[True, True, False]
    ).drop(columns=["decision_priority"])

    return result_df

def save_numeric_feature_selection(
    selection_df: pd.DataFrame,
    output_file: Path = PLOTS_DIR / "feature_selection_numeric.csv"
) -> None:
    selection_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nFeature selection fails saglabāts: {output_file}")


def run_full_eda(df: pd.DataFrame, label_col: str = "anomaly_label") -> None:
    create_plot_folders()
    show_dataset_info(df)
    check_duplicates(df, "Dataset")

    if label_col not in df.columns:
        print(f"Kolonna '{label_col}' netika atrasta datu kopā.")
        return

    numeric_features = get_numeric_features(df, exclude_columns=[label_col])
    binary_features = get_binary_features(df, exclude_columns=[label_col])

    numeric_features = [col for col in numeric_features if col not in binary_features]

    if not numeric_features and not binary_features:
        print("Nav atrasti atribūti analīzei.")
        return

    numeric_features_non_constant = get_non_constant_features(df, numeric_features)
    binary_features_non_constant = get_non_constant_features(df, binary_features)

    print(f"\nNumeric features: {len(numeric_features)}")
    print(f"Binary features: {len(binary_features)}")

    print("\nTiek veikta papildu atribūtu analīze...")
    numeric_analysis = analyze_numeric_features_kurtosis(df, numeric_features_non_constant)
    binary_analysis = analyze_binary_features_statistical(
        df=df,
        binary_features=binary_features_non_constant,
        exclude_columns=[label_col]
    )

    save_dataset_statistics(
        df=df,
        numeric_analysis=numeric_analysis,
        binary_analysis=binary_analysis
    )

    print("\nTiek veidoti grafiki...")
    plot_label_distribution(df, label_col=label_col)

    if numeric_features:
        plot_feature_histograms(df, numeric_features)
        plot_feature_boxplots(df, numeric_features)
        plot_boxplots_by_label(df, numeric_features, label_col=label_col)
        plot_histograms_by_label(df, numeric_features, label_col=label_col)

    corr_matrix = pd.DataFrame()

    if len(numeric_features_non_constant) >= 2:
        corr_matrix = plot_correlation_heatmap(df, numeric_features_non_constant)

        if not corr_matrix.empty:
            scatter_pairs = get_correlated_feature_pairs(corr_matrix)

            if scatter_pairs:
                plot_scatterplots(df, scatter_pairs, label_col=label_col)
                print(f"Scatter diagrammām atlasīti {len(scatter_pairs)} atribūtu pāri.")
            else:
                print("Nav atrasti atribūtu pāri ar pietiekamu korelāciju scatter diagrammām.")
        else:
            print("Korelācijas matricu neizdevās izveidot.")
    else:
        print("Korelācijas matricai un scatter diagrammām nepietiek nekonstantu skaitlisko atribūtu.")

    if numeric_features_non_constant:
        numeric_feature_selection_df = build_numeric_feature_selection(
        numeric_features=numeric_features_non_constant,
        numeric_analysis=numeric_analysis,
        corr_matrix=corr_matrix,
        kurtosis_threshold=3.0,
        correlation_threshold=0.7,
        max_group_size=3
    )
    save_numeric_feature_selection(numeric_feature_selection_df)

    if binary_features:
        plot_binary_countplots(df, binary_features, label_col=label_col)

if __name__ == "__main__":
    df = load_data(CLEANED_DATA)
    run_full_eda(df)