from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# PATHS
# =========================
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


# =========================
# LOAD DATA
# =========================
X_train = pd.read_csv(DATA_DIR / "train_data.csv")
X_val = pd.read_csv(DATA_DIR / "validation_data.csv")
X_test = pd.read_csv(DATA_DIR / "test_data.csv")

y_val = pd.read_csv(DATA_DIR / "validation_labels.csv")
y_test = pd.read_csv(DATA_DIR / "test_labels.csv")

y_val = y_val["anomaly_label"].astype(int).values
y_test = y_test["anomaly_label"].astype(int).values

# tensorflow works most cleanly with float32
X_train_np = X_train.astype("float32").values
X_val_np = X_val.astype("float32").values
X_test_np = X_test.astype("float32").values

input_dim = X_train_np.shape[1]


# =========================
# MODEL
# =========================
class Autoencoder(Model):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(8, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(8,)),
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


# =========================
# TRAIN MODEL
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_np,
    X_train_np,
    validation_data=(X_val_np, X_val_np),
    epochs=100,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)


# =========================
# VALIDATION SCORES
# =========================
# reconstruction error = anomaly score
val_reconstructions = model.predict(X_val_np, verbose=0)
val_scores = np.mean(np.square(X_val_np - val_reconstructions), axis=1)


# =========================
# THRESHOLD SEARCH
# =========================
thresholds = np.unique(val_scores)

best_f1 = -1.0
best_threshold = None
best_preds = None

for threshold in thresholds:
    val_preds = (val_scores >= threshold).astype(int)
    current_f1 = f1_score(y_val, val_preds, zero_division=0)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold
        best_preds = val_preds


# =========================
# VALIDATION METRICS
# =========================
val_precision = precision_score(y_val, best_preds, zero_division=0)
val_recall = recall_score(y_val, best_preds, zero_division=0)
val_accuracy = accuracy_score(y_val, best_preds)

val_roc_auc = roc_auc_score(y_val, val_scores)
val_pr_auc = average_precision_score(y_val, val_scores)

val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, best_preds).ravel()
val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0


# =========================
# SAVE VALIDATION RESULTS
# =========================
val_results = X_val.copy()
val_results["anomaly_score"] = val_scores
val_results["predicted_anomaly"] = best_preds
val_results["anomaly_label"] = y_val

val_results.to_csv(RESULTS_DIR / "ae_validation_scores.csv", index=False)


# =========================
# VALIDATION PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("Validation reconstruction error distribution")
plt.xlabel("Reconstruction error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ae_validation_scores_hist.png", dpi=300)
plt.close()


# =========================
# TRAINING LOSS PLOT
# =========================
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


# =========================
# TEST
# =========================
test_reconstructions = model.predict(X_test_np, verbose=0)
test_scores = np.mean(np.square(X_test_np - test_reconstructions), axis=1)

test_preds = (test_scores >= best_threshold).astype(int)


# =========================
# TEST METRICS
# =========================
test_precision = precision_score(y_test, test_preds, zero_division=0)
test_recall = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)
test_accuracy = accuracy_score(y_test, test_preds)

test_roc_auc = roc_auc_score(y_test, test_scores)
test_pr_auc = average_precision_score(y_test, test_scores)

test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_preds).ravel()
test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0.0


# =========================
# SAVE TEST RESULTS
# =========================
test_results = X_test.copy()
test_results["anomaly_score"] = test_scores
test_results["predicted_anomaly"] = test_preds
test_results["anomaly_label"] = y_test

test_results.to_csv(RESULTS_DIR / "ae_test_scores_predictions.csv", index=False)


# =========================
# SAVE METRICS
# =========================
metrics_df = pd.DataFrame([{
    "model": "Autoencoder",
    "selected_threshold": best_threshold,

    "val_precision": val_precision,
    "val_recall": val_recall,
    "val_f1": best_f1,
    "val_accuracy": val_accuracy,
    "val_specificity": val_specificity,
    "val_roc_auc": val_roc_auc,
    "val_pr_auc": val_pr_auc,
    "val_tn": val_tn,
    "val_fp": val_fp,
    "val_fn": val_fn,
    "val_tp": val_tp,

    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "test_accuracy": test_accuracy,
    "test_specificity": test_specificity,
    "test_roc_auc": test_roc_auc,
    "test_pr_auc": test_pr_auc,
    "test_tn": test_tn,
    "test_fp": test_fp,
    "test_fn": test_fn,
    "test_tp": test_tp
}])

metrics_file = RESULTS_DIR / "all_models_metrics.csv"

if metrics_file.exists():
    metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
else:
    metrics_df.to_csv(metrics_file, index=False)


# =========================
# FINISH
# =========================
print("done")