from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random 

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "train_data.csv")
X_val = pd.read_csv(DATA_DIR / "validation_data.csv")

y_val = pd.read_csv(DATA_DIR / "validation_labels.csv")
y_val = y_val["anomaly_label"].astype(int).values

X_train_np = X_train.astype("float32").values
X_val_np = X_val.astype("float32").values

input_dim = X_train_np.shape[1]

class Autoencoder(Model):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(4, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder(input_dim)

model.compile(
    optimizer="adam",
    loss="mse"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_np,
    X_train_np,
    validation_data=(X_val_np, X_val_np),
    epochs=128,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)

model.save_weights(MODELS_DIR / "autoencoder.weights.h5")

val_reconstructions = model.predict(X_val_np, verbose=0)
val_scores = np.mean(np.square(X_val_np - val_reconstructions), axis=1)


precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, val_scores)

precision_for_thr = precision_curve[:-1]
recall_for_thr = recall_curve[:-1]

f1_scores = 2 * (precision_for_thr * recall_for_thr) / (
    precision_for_thr + recall_for_thr + 1e-8
)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

best_preds = (val_scores >= best_threshold).astype(int)

val_precision = precision_score(y_val, best_preds, zero_division=0)
val_recall = recall_score(y_val, best_preds, zero_division=0)
val_f1 = f1_score(y_val, best_preds, zero_division=0)
val_accuracy = accuracy_score(y_val, best_preds)

val_roc_auc = roc_auc_score(y_val, val_scores)
val_pr_auc = average_precision_score(y_val, val_scores)

val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, best_preds).ravel()
val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0

val_results = X_val.copy()
val_results["anomaly_score"] = val_scores
val_results["predicted_anomaly"] = best_preds
val_results["anomaly_label"] = y_val

val_results.to_csv(RESULTS_DIR / "ae_validation_scores.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("Validation reconstruction error distribution")
plt.xlabel("Reconstruction error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ae_validation_scores_hist.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve)
plt.title("Autoencoder validation PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ae_validation_pr_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Autoencoder training history")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ae_training_history.png", dpi=300)
plt.close()

metrics_df = pd.DataFrame([{
    "model": "Autoencoder",
    "threshold_strategy": "PR_curve_max_F1",
    "selected_threshold": best_threshold,

    "val_precision": val_precision,
    "val_recall": val_recall,
    "val_f1": val_f1,
    "val_accuracy": val_accuracy,
    "val_specificity": val_specificity,
    "val_roc_auc": val_roc_auc,
    "val_pr_auc": val_pr_auc,
    "val_tn": val_tn,
    "val_fp": val_fp,
    "val_fn": val_fn,
    "val_tp": val_tp
}])

metrics_file = RESULTS_DIR / "validation_metrics.csv"

if metrics_file.exists():
    metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
else:
    metrics_df.to_csv(metrics_file, index=False)
