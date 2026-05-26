import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

from paths import CLEANED_DATA, PLOTS_EDA_DIR

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

# This EDA script is used to better understand the structure of the dataset before training anomaly detection models.
# For numerical features, the script checks distributions, boxplots, correlations and kurtosis values to see how the data behaves and
# whether there are possible outliers or heavy-tailed distributions.

# For binary features, variance and mutual information are calculated to understand if these attributes contain useful information or if they are almost constant and not very informative.
# Different visualizations are also generated to compare normal and anomalous observations and to see how features are distributed between classes.
# The main goal of this analysis is to identify potentially useful features, detect data quality problems and better prepare the data for anomaly detection experiments.


def create_plot_folders():
    for folder in PLOT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>| ]+', "_", str(name))


def load_data(path):
    return pd.read_csv(path)

def get_numeric_features(df, exclude=None):
    if exclude is None:
        exclude = []

    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    result = []

    for col in cols:
        if col not in exclude:
            result.append(col)

    return result


def get_binary_features(df, exclude=None):
    if exclude is None:
        exclude = []

    binary = []

    for col in df.columns:
        if col in exclude:
            continue

        vals = df[col].dropna().unique()

        if len(vals) <= 2:
            if set(vals).issubset({0, 1}):
                binary.append(col)

    return binary


def get_non_constant_features(df, features):
    result = []

    for col in features:
        if df[col].nunique(dropna=True) > 1:
            result.append(col)

    return result


def compute_kurtosis(series):
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) == 0:
        return np.nan

    if series.nunique() <= 1:
        return np.nan

    std = series.std(ddof=0)

    if std == 0:
        return np.nan

    z = (series - series.mean()) / std

    return float(np.mean(z ** 4))


def analyze_numeric_features(df, numeric_features):
    results = {}

    for feature in numeric_features:
        k = compute_kurtosis(df[feature])

        if pd.isna(k):
            text = "not_applicable"
        elif k > 3:
            text = "high_tail_weight"
        else:
            text = "normal_or_light"

        results[feature] = {
            "pearson_kurtosis": k,
            "interpretation": text
        }

    return results


def analyze_binary_features(df, binary_features):
    results = {}

    for feature in binary_features:
        try:
            s = df[feature].dropna()

            if s.nunique() < 2:
                results[feature] = {
                    "variance": np.nan,
                    "mean_mi": np.nan
                }
                continue

            p = s.mean()
            variance = p * (1 - p)

            predictor_cols = []

            for col in df.columns:
                if col != feature:
                    predictor_cols.append(col)

            X = df[predictor_cols].copy()
            y = df[feature].copy()

            mask = y.notna()

            X = X.loc[mask]
            y = y.loc[mask].astype(int)

            X = pd.get_dummies(X, drop_first=True)

            bool_cols = X.select_dtypes(include=["bool"]).columns

            for col in bool_cols:
                X[col] = X[col].astype(int)

            X = X.select_dtypes(include=[np.number])

            if len(X.columns) == 0:
                mean_mi = np.nan
            else:
                X = X.fillna(X.median(numeric_only=True))

                discrete_mask = []

                for col in X.columns:
                    if pd.api.types.is_integer_dtype(X[col]) and X[col].nunique() <= 10:
                        discrete_mask.append(True)
                    else:
                        discrete_mask.append(False)
                mi_scores = mutual_info_classif(
                    X=X,
                    y=y,
                    discrete_features=discrete_mask,
                    random_state=42
                )
                mean_mi = float(np.mean(mi_scores))
            results[feature] = {
                "variance": variance,
                "mean_mi": mean_mi
            }
        except:
            results[feature] = {
                "variance": np.nan,
                "mean_mi": np.nan
            }

    return results


def save_dataset_statistics(df, numeric_analysis=None, binary_analysis=None):
    if numeric_analysis is None:
        numeric_analysis = {}

    if binary_analysis is None:
        binary_analysis = {}

    rows = []

    for column in df.columns:
        s = df[column]
        clean = s.dropna()

        row = {
            "feature": column,
            "dtype": str(s.dtype),
            "missing_count": s.isnull().sum(),
            "unique_count": clean.nunique()
        }

        if len(clean) > 0:
            row["mode"] = clean.mode().iloc[0]

        if pd.api.types.is_numeric_dtype(s):
            row["min"] = clean.min()
            row["max"] = clean.max()
            row["mean"] = clean.mean()
            row["median"] = clean.median()
            row["std"] = clean.std()

        if column in numeric_analysis:
            row["pearson_kurtosis"] = numeric_analysis[column]["pearson_kurtosis"]
            row["kurtosis_info"] = numeric_analysis[column]["interpretation"]

        if column in binary_analysis:
            row["variance"] = binary_analysis[column]["variance"]
            row["mean_mi"] = binary_analysis[column]["mean_mi"]

        rows.append(row)

    stats_df = pd.DataFrame(rows)

    stats_df.to_csv(
        PLOTS_DIR / "dataset_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )


def plot_label_distribution(df, label_col="anomaly_label"):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=label_col)
    plt.tight_layout()
    plt.savefig(
        PLOT_FOLDERS["label_distribution"] / "label_distribution.png",
        dpi=300
    )

    plt.close()


def plot_feature_histograms(df, features):
    for feature in features:
        data = df[feature].dropna()

        if len(data) == 0:
            continue
        plt.figure(figsize=(8, 5))
        sns.histplot(data, kde=True)
        plt.title(feature)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["feature_histograms"] /
            f"{sanitize_filename(feature)}_hist.png",
            dpi=300
        )

        plt.close()


def plot_feature_boxplots(df, features):
    for feature in features:
        data = df[feature].dropna()

        if len(data) == 0:
            continue
        plt.figure(figsize=(7, 5))
        sns.boxplot(y=data)
        plt.title(feature)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["feature_boxplots"] /
            f"{sanitize_filename(feature)}_box.png",
            dpi=300
        )

        plt.close()


def plot_boxplots_by_label(df, features, label_col="anomaly_label"):
    for feature in features:
        temp = df[[feature, label_col]].dropna()

        if len(temp) == 0:
            continue
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=temp, x=label_col, y=feature)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["boxplots_by_label"] /
            f"{sanitize_filename(feature)}_by_label.png",
            dpi=300
        )

        plt.close()


def plot_histograms_by_label(df, features, label_col="anomaly_label"):
    for feature in features:
        temp = df[[feature, label_col]].dropna()

        if len(temp) == 0:
            continue

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=temp,
            x=feature,
            hue=label_col,
            kde=True,
            element="step"
        )
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["histograms_by_label"] /
            f"{sanitize_filename(feature)}_hist_by_label.png",
            dpi=300
        )

        plt.close()


def plot_binary_countplots(df, features, label_col="anomaly_label"):
    for feature in features:
        temp = df[[feature, label_col]].dropna()

        if len(temp) == 0:
            continue

        plt.figure(figsize=(7, 5))

        sns.countplot(data=temp, x=feature, hue=label_col)

        plt.tight_layout()

        plt.savefig(
            PLOT_FOLDERS["categorical_countplots"] /
            f"{sanitize_filename(feature)}_countplot.png",
            dpi=300
        )

        plt.close()


def plot_correlation_heatmap(df, numeric_features):
    if len(numeric_features) < 2:
        return pd.DataFrame()

    corr = df[numeric_features].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    sns.heatmap(corr, annot=False, cmap="coolwarm")

    plt.tight_layout()

    plt.savefig(
        PLOT_FOLDERS["correlation"] / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    corr.to_csv(
        PLOT_FOLDERS["correlation"] / "correlation_matrix.csv",
        encoding="utf-8-sig"
    )

    return corr


def get_correlated_feature_pairs(corr_matrix, threshold=0.7):
    pairs = []

    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1 = cols[i]
            f2 = cols[j]

            corr = corr_matrix.loc[f1, f2]

            if pd.notna(corr):
                if abs(corr) >= threshold:
                    pairs.append((f1, f2, corr))

    return pairs


def plot_scatterplots(df, pairs, label_col="anomaly_label"):
    for f1, f2, corr in pairs:
        temp = df[[f1, f2, label_col]].dropna()

        if len(temp) == 0:
            continue

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=temp,
            x=f1,
            y=f2,
            hue=label_col,
            alpha=0.7
        )
        plt.tight_layout()

        plt.savefig(
            PLOT_FOLDERS["scatterplots"] /
            f"{sanitize_filename(f1)}_{sanitize_filename(f2)}.png",
            dpi=300
        )

        plt.close()


def run_full_eda(df, label_col="anomaly_label"):
    create_plot_folders()
    if label_col not in df.columns:
        return
    numeric_features = get_numeric_features(df, exclude=[label_col])

    binary_features = get_binary_features(df, exclude=[label_col])

    numeric_features = [
        col for col in numeric_features
        if col not in binary_features
    ]
    numeric_non_constant = get_non_constant_features(
        df,
        numeric_features
    )
    binary_non_constant = get_non_constant_features(
        df,
        binary_features
    )
    numeric_analysis = analyze_numeric_features(
        df,
        numeric_non_constant
    )
    binary_analysis = analyze_binary_features(
        df,
        binary_non_constant
    )
    save_dataset_statistics(
        df,
        numeric_analysis,
        binary_analysis
    )

    plot_label_distribution(df, label_col)

    if len(numeric_features) > 0:
        plot_feature_histograms(df, numeric_features)

        plot_feature_boxplots(df, numeric_features)

        plot_boxplots_by_label(
            df,
            numeric_features,
            label_col
        )

        plot_histograms_by_label(
            df,
            numeric_features,
            label_col
        )

    corr_matrix = pd.DataFrame()

    if len(numeric_non_constant) >= 2:
        corr_matrix = plot_correlation_heatmap(
            df,
            numeric_non_constant
        )

        if len(corr_matrix) > 0:
            pairs = get_correlated_feature_pairs(corr_matrix)

            if len(pairs) > 0:
                plot_scatterplots(
                    df,
                    pairs,
                    label_col
                )

    if len(binary_features) > 0:
        plot_binary_countplots(
            df,
            binary_features,
            label_col
        )

if __name__ == "__main__":
    df = load_data(CLEANED_DATA)
    run_full_eda(df)