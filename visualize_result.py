from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "final_visualizations"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "isolationforest": "Isolation Forest",
    "oneclasssvm": "One-Class SVM",
    "localoutlierfactor": "Local Outlier Factor",
    "autoencoder": "Autoencoder",
    "variational_autoencoder": "Variational Autoencoder"
}

MODEL_COLORS = {
    "Isolation Forest": "#4C72B0",
    "One-Class SVM": "#DD8452",
    "Local Outlier Factor": "#55A868",
    "Autoencoder": "#C44E52",
    "Variational Autoencoder": "#8172B2"
}


def load_model_results(model_key):
    file_path = RESULTS_DIR / f"{model_key}_test_scores_predictions.csv"
    if not file_path.exists():
        print(f"[WARN] {file_path} not found")
        return None
    return pd.read_csv(file_path)


def get_features(df):
    return df.drop(columns=[
        "anomaly_score",
        "predicted_anomaly",
        "anomaly_label",
        "cell_type",
        "disease_category"
    ], errors="ignore")


def plot_scatter(df, model_name):
    X = get_features(df)

    if X.shape[1] < 2:
        print(f"[WARN] {model_name}: not enough features for PCA")
        return

    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X)

    df_plot = pd.DataFrame({
        "PC1": comps[:, 0],
        "PC2": comps[:, 1],
        "pred": df["predicted_anomaly"].astype(int),
        "true": df["anomaly_label"].astype(int)
    })

    explained_var = pca.explained_variance_ratio_.sum()

    # Predicted anomalies
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df_plot[df_plot["pred"] == 0]["PC1"],
        df_plot[df_plot["pred"] == 0]["PC2"],
        s=15,
        alpha=0.45,
        color="steelblue",
        label="Normal"
    )
    plt.scatter(
        df_plot[df_plot["pred"] == 1]["PC1"],
        df_plot[df_plot["pred"] == 1]["PC2"],
        s=22,
        alpha=0.9,
        color="red",
        label="Anomaly"
    )
    plt.title(f"{model_name} – Predicted anomalies\nExplained variance = {explained_var:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_scatter_pred.png", dpi=300)
    plt.close()

    # True anomalies
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df_plot[df_plot["true"] == 0]["PC1"],
        df_plot[df_plot["true"] == 0]["PC2"],
        s=15,
        alpha=0.45,
        color="steelblue",
        label="Normal"
    )
    plt.scatter(
        df_plot[df_plot["true"] == 1]["PC1"],
        df_plot[df_plot["true"] == 1]["PC2"],
        s=22,
        alpha=0.9,
        color="red",
        label="Anomaly"
    )
    plt.title(f"{model_name} – True anomalies\nExplained variance = {explained_var:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_scatter_true.png", dpi=300)
    plt.close()


def normalize_model_name(name: str) -> str:
    if name == "IsolationForest":
        return "Isolation Forest"
    if name == "OCSVM":
        return "One-Class SVM"
    if name == "LOF":
        return "Local Outlier Factor"
    if name == "Autoencoder":
        return "Autoencoder"
    if name == "VAE":
        return "Variational Autoencoder"
    if name == "VAE_64_32_8_bs16":
        return "Variational Autoencoder"
    return name


def plot_metrics_comparison():
    metrics_file = RESULTS_DIR / "test_metrics.csv"
    if not metrics_file.exists():
        print("[WARN] test_metrics.csv not found")
        return

    df = pd.read_csv(metrics_file)
    df["model"] = df["model"].apply(normalize_model_name)

    df = df.drop_duplicates(subset="model", keep="last")

    colors = [MODEL_COLORS.get(model, "gray") for model in df["model"]]

    # F1
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["test_f1"], color=colors)
    plt.title("Test F1-score comparison")
    plt.ylabel("F1-score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_f1.png", dpi=300)
    plt.close()

    # ROC AUC
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["test_roc_auc"], color=colors)
    plt.title("Test ROC AUC comparison")
    plt.ylabel("ROC AUC")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_roc_auc.png", dpi=300)
    plt.close()

    # PR AUC
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["test_pr_auc"], color=colors)
    plt.title("Test PR AUC comparison")
    plt.ylabel("PR AUC")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_pr_auc.png", dpi=300)
    plt.close()


for key, name in MODELS.items():
    df = load_model_results(key)
    if df is None:
        continue

    print(f"Processing {name}")
    plot_scatter(df, name)

plot_metrics_comparison()

print("done")