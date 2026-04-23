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
    confusion_matrix,
    precision_recall_curve
)
from tensorflow.keras import Model, layers
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
# load preprocessed datasets (scaling already done earlier if needed)
X_train = pd.read_csv(DATA_DIR / "train_data.csv")
X_val = pd.read_csv(DATA_DIR / "validation_data.csv")
X_test = pd.read_csv(DATA_DIR / "test_data.csv")

y_val = pd.read_csv(DATA_DIR / "validation_labels.csv")
y_test = pd.read_csv(DATA_DIR / "test_labels.csv")

y_val = y_val["anomaly_label"].astype(int).values
y_test = y_test["anomaly_label"].astype(int).values

X_train_np = X_train.astype("float32").values
X_val_np = X_val.astype("float32").values
X_test_np = X_test.astype("float32").values

input_dim = X_train_np.shape[1]


# =========================
# VAE MODEL
# =========================
class VAE(Model):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder base
        self.encoder_net = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu")
        ])

        # latent parameters
        self.mu_layer = layers.Dense(latent_dim)
        self.log_var_layer = layers.Dense(latent_dim)

        # decoder
        self.decoder_net = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])

    def sample(self, mu, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    def encode(self, x):
        x_encoded = self.encoder_net(x)
        mu = self.mu_layer(x_encoded)
        log_var = self.log_var_layer(x_encoded)
        return mu, log_var

    def decode(self, z):
        return self.decoder_net(z)

    def call(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        reconstructed = self.decode(z)

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        self.add_loss(kl_loss)

        return reconstructed


model = VAE(input_dim=input_dim, latent_dim=8)

model.compile(
    optimizer="adam",
    loss="mse"
)


# =========================
# TRAIN MODEL
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)

history = model.fit(
    X_train_np,
    X_train_np,
    validation_data=(X_val_np, X_val_np),
    epochs=100,
    batch_size=16,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)


# =========================
# VALIDATION SCORES
# =========================
# anomaly score = reconstruction error
val_reconstructions = model.predict(X_val_np, verbose=0)
val_scores = np.mean(np.square(X_val_np - val_reconstructions), axis=1)


# =========================
# THRESHOLD SEARCH VIA PR CURVE
# =========================
precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, val_scores)

# precision_recall_curve returns one more precision/recall value than thresholds
precision_for_f1 = precision_curve[:-1]
recall_for_f1 = recall_curve[:-1]

f1_scores = 2 * (precision_for_f1 * recall_for_f1) / (
    precision_for_f1 + recall_for_f1 + 1e-8
)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

best_preds = (val_scores >= best_threshold).astype(int)


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

val_results.to_csv(RESULTS_DIR / "vae_validation_scores.csv", index=False)


# =========================
# VALIDATION PLOTS
# =========================
plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("Validation VAE reconstruction error distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vae_validation_scores_hist.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve)
plt.title("VAE validation PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vae_validation_pr_curve.png", dpi=300)
plt.close()


# =========================
# TRAINING HISTORY PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("VAE training history")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vae_training_history.png", dpi=300)
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

test_results.to_csv(RESULTS_DIR / "vae_test_scores_predictions.csv", index=False)


# =========================
# TEST PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.hist(test_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("Test VAE reconstruction error distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vae_test_scores_hist.png", dpi=300)
plt.close()


# =========================
# SAVE METRICS
# =========================
metrics_df = pd.DataFrame([{
    "model": "VAE_64_32_8_bs16",
    "threshold_strategy": "PR_curve_max_F1",
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