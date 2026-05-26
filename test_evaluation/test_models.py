import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import time
from paths import (
    TEST_DATA,
    TEST_LABELS,
    RESULTS_DIR,
    MODELS_DIR
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix
)
from tensorflow.keras import Model, layers

# This script evaluates anomaly detection models on the test dataset.
# The saved thresholds from the validation stage are loaded first, and then each model predicts anomaly scores for the test data.
# After prediction, the script calculates evaluation metrics such as precision, recall, F1-score and PR AUC to compare model performance on unseen data.
# Histograms of anomaly scores are also generated to visualize score distributions and threshold positions.

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(TEST_DATA)
y_test = pd.read_csv(TEST_LABELS)

y_test = y_test["anomaly_label"].astype(int).values
X_test_np = X_test.astype("float32").values

input_dim = X_test_np.shape[1]

validation_metrics = pd.read_csv(RESULTS_DIR / "validation_metrics.csv")

def get_threshold(model_name):
    rows = validation_metrics[validation_metrics["model"] == model_name]

    if rows.empty:
        raise ValueError(f"No threshold found for model: {model_name}")

    return rows.iloc[-1]["selected_threshold"]

def calculate_metrics(model_name, test_scores, threshold, execution_time):
    test_preds = (test_scores >= threshold).astype(int)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    test_pr_auc = average_precision_score(y_test, test_scores)

    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_preds).ravel()
    
    test_results = X_test.copy()
    test_results["anomaly_score"] = test_scores
    test_results["predicted_anomaly"] = test_preds
    test_results["anomaly_label"] = y_test

    file_name = model_name.lower().replace(" ", "_").replace("-", "_")
    test_results.to_csv(RESULTS_DIR / f"{file_name}_test_scores_predictions.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(test_scores, bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.title(f"{model_name} test anomaly score distribution")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{file_name}_test_scores_hist.png", dpi=300)
    plt.close()

    return {
        "model": model_name,
        "threshold_strategy": "PR_curve_max_F1",
        "selected_threshold": threshold,

        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_pr_auc": test_pr_auc,
        "test_execution_time_sec": execution_time,
        "test_tn": test_tn,
        "test_fp": test_fp,
        "test_fn": test_fn,
        "test_tp": test_tp
    }

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

class VAE(Model):
    def __init__(self, input_dim, latent_dim=4):
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

    def encode(self, x):
        x_encoded = self.encoder_net(x)
        mu = self.mu_layer(x_encoded)
        log_var = self.log_var_layer(x_encoded)
        return mu, log_var

    def decode(self, z):
        return self.decoder_net(z)

    def call(self, x):
        mu, log_var = self.encode(x)
        reconstructed = self.decode(mu)
        return reconstructed


def reconstruct_vae(model, X):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    mu, log_var = model.encode(X_tensor)
    reconstructed = model.decode(mu)
    return reconstructed.numpy()

all_metrics = []


model_name = "IsolationForest"
threshold = get_threshold(model_name)

if_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
start_time = time.perf_counter()

test_scores = -if_model.decision_function(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time

all_metrics.append(calculate_metrics(model_name, test_scores, threshold, execution_time))

model_name = "LocalOutlierFactor"
threshold = get_threshold(model_name)

lof_model = joblib.load(MODELS_DIR / "lof.pkl")
start_time = time.perf_counter()

test_scores = -lof_model.decision_function(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time

all_metrics.append(calculate_metrics(model_name, test_scores, threshold, execution_time))

model_name = "OneClassSVM"
threshold = get_threshold(model_name)

ocsvm_model = joblib.load(MODELS_DIR / "ocsvm.pkl")
start_time = time.perf_counter()

test_scores = -ocsvm_model.decision_function(X_test)

end_time = time.perf_counter()
execution_time = end_time - start_time

all_metrics.append(calculate_metrics(model_name, test_scores, threshold, execution_time))

model_name = "Autoencoder"
threshold = get_threshold(model_name)

ae_model = Autoencoder(input_dim)
ae_model(tf.zeros((1, input_dim)))
ae_model.load_weights(MODELS_DIR / "autoencoder.weights.h5")
start_time = time.perf_counter()

test_reconstructions = ae_model.predict(X_test_np, verbose=0)
test_scores = np.mean(np.square(X_test_np - test_reconstructions), axis=1)

end_time = time.perf_counter()
execution_time = end_time - start_time

all_metrics.append(calculate_metrics(model_name, test_scores, threshold, execution_time))

model_name = "Variational Autoencoder"
threshold = get_threshold(model_name)

vae_model = VAE(input_dim=input_dim, latent_dim=4)
vae_model(tf.zeros((1, input_dim)))
vae_model.load_weights(MODELS_DIR / "vae.weights.h5")
start_time = time.perf_counter()

test_reconstructions = reconstruct_vae(vae_model, X_test_np)
test_scores = np.mean(np.square(X_test_np - test_reconstructions), axis=1)

end_time = time.perf_counter()
execution_time = end_time - start_time

all_metrics.append(calculate_metrics(model_name, test_scores, threshold, execution_time))

test_metrics_df = pd.DataFrame(all_metrics)
test_metrics_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)

print("Final test evaluation done")
print(test_metrics_df)