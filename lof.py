from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
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

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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


# =========================
# TRAIN MODEL
# =========================
# novelty=True is required to evaluate unseen validation/test data
model = LocalOutlierFactor(
    n_neighbors=100,
    contamination=0.2,
    novelty=True,
    n_jobs=-1
)

model.fit(X_train)


# =========================
# VALIDATION SCORES
# =========================
# for LOF, decision_function gives larger values for more normal samples
# invert sign so that higher score = more anomalous
val_scores = -model.decision_function(X_val)


# =========================
# THRESHOLD SEARCH VIA PR CURVE
# =========================
precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, val_scores)

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

val_results.to_csv(RESULTS_DIR / "lof_validation_scores.csv", index=False)


# =========================
# VALIDATION PLOTS
# =========================
plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("LOF validation score distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lof_validation_scores_hist.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve)
plt.title("LOF validation PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lof_validation_pr_curve.png", dpi=300)
plt.close()


# =========================
# TEST
# =========================
test_scores = -model.decision_function(X_test)
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

test_results.to_csv(RESULTS_DIR / "lof_test_scores_predictions.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(test_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("LOF test score distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lof_test_scores_hist.png", dpi=300)
plt.close()


# =========================
# SAVE METRICS
# =========================
metrics_df = pd.DataFrame([{
    "model": "LocalOutlierFactor",
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