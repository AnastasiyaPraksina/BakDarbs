import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random 
from paths import (
    TRAIN_DATA,
    VALIDATION_DATA,
    VALIDATION_LABELS,
    RESULTS_DIR,
    MODELS_DIR
)
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(TRAIN_DATA)
X_val = pd.read_csv(VALIDATION_DATA)

y_val = pd.read_csv(VALIDATION_LABELS)
y_val = y_val["anomaly_label"].astype(int).values
X_train_np = X_train.astype("float32").values
X_val_np = X_val.astype("float32").values

input_dim = X_train_np.shape[1]

class VAE(Model):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_net = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu")
        ])

        self.mu_layer = layers.Dense(latent_dim)
        self.log_var_layer = layers.Dense(latent_dim)

        self.decoder_net = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])

    def sample(self, mu, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mu), seed=42)
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

        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        self.add_loss(kl_loss)

        return reconstructed


model = VAE(input_dim=input_dim, latent_dim=4)

model.compile(
    optimizer="adam",
    loss="mse"
)


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

model.save_weights(MODELS_DIR / "vae.weights.h5")


def reconstruct_deterministic(model, X):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    mu, log_var = model.encode(X_tensor)
    reconstructed = model.decode(mu)
    return reconstructed.numpy()

val_reconstructions = reconstruct_deterministic(model, X_val_np)
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

val_results.to_csv(RESULTS_DIR / "vae_validation_scores.csv", index=False)


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


metrics_df = pd.DataFrame([{
    "model": "Variational Autoencoder",
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