from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


sns.set_theme(style="whitegrid")

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots_split_data")

TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
VALIDATION_DATA_FILE = DATA_DIR / "validation_data.csv"
TEST_DATA_FILE = DATA_DIR / "test_data.csv"

TRAIN_LABELS_FILE = DATA_DIR / "train_labels.csv"
VALIDATION_LABELS_FILE = DATA_DIR / "validation_labels.csv"
TEST_LABELS_FILE = DATA_DIR / "test_labels.csv"

MAX_FEATURES_TO_PLOT = 12
RANDOM_STATE = 42

PLOT_FOLDERS = {
    "label_distribution": PLOTS_DIR / "label_distribution",
    "feature_histograms": PLOTS_DIR / "feature_histograms",
    "feature_boxplots": PLOTS_DIR / "feature_boxplots",
    "histograms_by_label": PLOTS_DIR / "histograms_by_label",
    "boxplots_by_label": PLOTS_DIR / "boxplots_by_label",
    "correlation": PLOTS_DIR / "correlation",
    "pca": PLOTS_DIR / "pca",
}


def create_plot_folders() -> None:
    for folder in PLOT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def load_split_data(
    data_file: Path,
    labels_file: Path,
    split_name: str
) -> pd.DataFrame:
    X = pd.read_csv(data_file)
    y = pd.read_csv(labels_file)

    df = X.copy()
    df["anomaly_label"] = y["anomaly_label"].astype(int)

    if "cell_type" in y.columns:
        df["cell_type"] = y["cell_type"]
    if "disease_category" in y.columns:
        df["disease_category"] = y["disease_category"]

    df["split"] = split_name
    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    excluded = {"anomaly_label"}
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_columns if col not in excluded]


def select_features_for_plotting(df: pd.DataFrame, max_features: int = MAX_FEATURES_TO_PLOT) -> list[str]:
    numeric_features = get_numeric_features(df)

    if not numeric_features:
        return []

    variances = df[numeric_features].var(numeric_only=True).sort_values(ascending=False)
    selected_features = variances.head(max_features).index.tolist()
    return selected_features


def plot_label_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    summary = pd.DataFrame({
        "split": ["Train", "Train", "Validation", "Validation", "Test", "Test"],
        "anomaly_label": [0, 1, 0, 1, 0, 1],
        "count": [
            (train_df["anomaly_label"] == 0).sum(),
            (train_df["anomaly_label"] == 1).sum(),
            (val_df["anomaly_label"] == 0).sum(),
            (val_df["anomaly_label"] == 1).sum(),
            (test_df["anomaly_label"] == 0).sum(),
            (test_df["anomaly_label"] == 1).sum(),
        ]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x="split", y="count", hue="anomaly_label")
    plt.title("Class distribution across dataset splits")
    plt.xlabel("Dataset split")
    plt.ylabel("Count")
    plt.legend(title="anomaly_label")
    plt.tight_layout()
    plt.savefig(PLOT_FOLDERS["label_distribution"] / "class_distribution_across_splits.png", dpi=300)
    plt.close()


def plot_feature_histograms(df: pd.DataFrame, split_name: str, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f"{split_name}: histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["feature_histograms"] / f"{split_name.lower()}_{sanitize_filename(feature)}_hist.png",
            dpi=300
        )
        plt.close()


def plot_feature_boxplots(df: pd.DataFrame, split_name: str, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=df[feature])
        plt.title(f"{split_name}: boxplot of {feature}")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["feature_boxplots"] / f"{split_name.lower()}_{sanitize_filename(feature)}_box.png",
            dpi=300
        )
        plt.close()


def plot_histograms_by_label(df: pd.DataFrame, split_name: str, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue="anomaly_label",
            bins=30,
            kde=True,
            stat="count",
            common_norm=False,
            element="step"
        )
        plt.title(f"{split_name}: histogram of {feature} by anomaly label")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["histograms_by_label"] / f"{split_name.lower()}_{sanitize_filename(feature)}_by_label_hist.png",
            dpi=300
        )
        plt.close()


def plot_boxplots_by_label(df: pd.DataFrame, split_name: str, features: list[str]) -> None:
    for feature in features:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x="anomaly_label", y=feature)
        plt.title(f"{split_name}: boxplot of {feature} by anomaly label")
        plt.xlabel("anomaly_label")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(
            PLOT_FOLDERS["boxplots_by_label"] / f"{split_name.lower()}_{sanitize_filename(feature)}_by_label_box.png",
            dpi=300
        )
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, split_name: str) -> None:
    numeric_features = get_numeric_features(df)

    if len(numeric_features) < 2:
        return

    corr = df[numeric_features].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"{split_name}: correlation heatmap")
    plt.tight_layout()
    plt.savefig(
        PLOT_FOLDERS["correlation"] / f"{split_name.lower()}_correlation_heatmap.png",
        dpi=300
    )
    plt.close()


def plot_pca_scatter(df: pd.DataFrame, split_name: str) -> None:
    numeric_features = get_numeric_features(df)

    if len(numeric_features) < 2:
        return

    X = df[numeric_features].copy()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "anomaly_label": df["anomaly_label"].values
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="anomaly_label",
        alpha=0.7
    )
    plt.title(
        f"{split_name}: PCA projection\n"
        f"Explained variance = {pca.explained_variance_ratio_.sum():.2%}"
    )
    plt.tight_layout()
    plt.savefig(
        PLOT_FOLDERS["pca"] / f"{split_name.lower()}_pca_scatter.png",
        dpi=300
    )
    plt.close()


def print_basic_info(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n=== {name} split ===")
        print(f"Shape: {df.shape}")
        print(f"Anomaly rate: {df['anomaly_label'].mean():.4f}")
        print(df["anomaly_label"].value_counts().sort_index())


def visualize_split(df: pd.DataFrame, split_name: str, features: list[str]) -> None:
    plot_feature_histograms(df, split_name, features)
    plot_feature_boxplots(df, split_name, features)
    plot_histograms_by_label(df, split_name, features)
    plot_boxplots_by_label(df, split_name, features)
    plot_pca_scatter(df, split_name)


create_plot_folders()

train_df = load_split_data(TRAIN_DATA_FILE, TRAIN_LABELS_FILE, "Train")
val_df = load_split_data(VALIDATION_DATA_FILE, VALIDATION_LABELS_FILE, "Validation")
test_df = load_split_data(TEST_DATA_FILE, TEST_LABELS_FILE, "Test")

print_basic_info(train_df, val_df, test_df)

plot_label_distribution(train_df, val_df, test_df)

selected_features = select_features_for_plotting(train_df, MAX_FEATURES_TO_PLOT)

print("\n=== Selected features for visualization ===")
print(selected_features)

visualize_split(train_df, "Train", selected_features)
visualize_split(val_df, "Validation", selected_features)
visualize_split(test_df, "Test", selected_features)

plot_correlation_heatmap(train_df, "Train")

print("\n=== Visualizations saved successfully ===")
for name, folder in PLOT_FOLDERS.items():
    print(f"{name}: {folder}")