import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from paths import RESULTS_DIR
import numpy as np

PLOTS_DIR = RESULTS_DIR / "final_visualizations"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "isolationforest": "Isolation Forest",
    "localoutlierfactor": "Local Outlier Factor",
    "oneclasssvm": "One-Class SVM",
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

def plot_f1_pr_auc_comparison():
    metrics_file = RESULTS_DIR / "test_metrics.csv"

    if not metrics_file.exists():
        print("[WARN] test_metrics.csv not found")
        return

    df = pd.read_csv(metrics_file)

    model_names = {
        "IsolationForest": "Izolācijas\nmeži",
        "LocalOutlierFactor": "LOF",
        "OneClassSVM": "Vienas klases\nSVM",
        "Autoencoder": "Autoenkoderis",
        "Variational Autoencoder": "Variacionālais\nautoenkoderis"
    }

    df["model_label"] = df["model"].map(model_names).fillna(df["model"])

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(x - width / 2, df["test_f1"], width, label="F1-rādītājs")
    bars2 = plt.bar(x + width / 2, df["test_pr_auc"], width, label="PR AUC")

    plt.title("Algoritmu veiktspējas salīdzinājums pēc F1-rādītāja un PR AUC", fontsize=16)
    plt.ylabel("Metrikas vērtība", fontsize=12)
    plt.xticks(x, df["model_label"], fontsize=11)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.tight_layout()

    output_path = PLOTS_DIR / "f1_pr_auc_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafiks saglabāts: {output_path}")

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

def plot_confusion_matrix(df, model_name):
    y_true = df["anomaly_label"].astype(int)
    y_pred = df["predicted_anomaly"].astype(int)

    plt.figure(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

def normalize_model_name(name: str) -> str:
    mapping = {
        "IsolationForest": "Isolation Forest",
        "LocalOutlierFactor": "Local Outlier Factor",
        "OneClassSVM": "One-Class SVM",
        "Autoencoder": "Autoencoder",
        "Variational Autoencoder": "Variational Autoencoder"
    }

    return mapping.get(name, name)

for key, name in MODELS.items():
    df = load_model_results(key)
    if df is None:
        continue

    print(f"Processing {name}")
    plot_scatter(df, name)
    plot_confusion_matrix(df, name)
    plot_f1_pr_auc_comparison()