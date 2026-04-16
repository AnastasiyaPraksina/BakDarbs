from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# LOAD DATA
# =========================
# load preprocessed datasets (scaling already done earlier if needed)
X_train = pd.read_csv(DATA_DIR / "train_data.csv")
X_val = pd.read_csv(DATA_DIR / "validation_data.csv")
X_test = pd.read_csv(DATA_DIR / "test_data.csv")

# load labels only for evaluation (not used in training)
y_val = pd.read_csv(DATA_DIR / "validation_labels.csv")
y_test = pd.read_csv(DATA_DIR / "test_labels.csv")

y_val = y_val["anomaly_label"].astype(int).values
y_test = y_test["anomaly_label"].astype(int).values


# =========================
# TRAIN MODEL
# =========================
# One-Class SVM is sensitive to scaling, so using already scaled data is important
model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.1
)

model.fit(X_train)


# =========================
# VALIDATION SCORES
# =========================
# decision_function is positive for inliers and negative for outliers
# invert sign so that higher score = more anomalous
val_scores = -model.decision_function(X_val)


# =========================
# THRESHOLD SEARCH
# =========================
# use all unique score values as candidate thresholds
thresholds = np.unique(val_scores)

best_f1 = -1.0
best_threshold = None
best_preds = None

for threshold in thresholds:
    # convert scores into binary anomaly labels
    val_preds = (val_scores >= threshold).astype(int)

    # compute F1-score (main metric for threshold selection)
    current_f1 = f1_score(y_val, val_preds, zero_division=0)

    # keep best threshold based on F1-score
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold
        best_preds = val_preds


# =========================
# VALIDATION METRICS
# =========================
# threshold-dependent metrics
val_precision = precision_score(y_val, best_preds, zero_division=0)
val_recall = recall_score(y_val, best_preds, zero_division=0)
val_accuracy = accuracy_score(y_val, best_preds)

# threshold-independent metrics
val_roc_auc = roc_auc_score(y_val, val_scores)
val_pr_auc = average_precision_score(y_val, val_scores)

# confusion matrix and specificity
val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, best_preds).ravel()
val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0


# =========================
# SAVE VALIDATION RESULTS
# =========================
# save validation scores and predictions for analysis
val_results = X_val.copy()
val_results["anomaly_score"] = val_scores
val_results["predicted_anomaly"] = best_preds
val_results["anomaly_label"] = y_val

val_results.to_csv(RESULTS_DIR / "ocsvm_validation_scores.csv", index=False)


# =========================
# VALIDATION PLOT
# =========================
# visualize score distribution and selected threshold
plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("Validation anomaly score distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ocsvm_validation_scores_hist.png", dpi=300)
plt.close()


# =========================
# TEST
# =========================
# compute anomaly scores for test set
test_scores = -model.decision_function(X_test)

# apply threshold selected on validation set
test_preds = (test_scores >= best_threshold).astype(int)


# =========================
# TEST METRICS
# =========================
# threshold-dependent metrics
test_precision = precision_score(y_test, test_preds, zero_division=0)
test_recall = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)
test_accuracy = accuracy_score(y_test, test_preds)

# threshold-independent metrics
test_roc_auc = roc_auc_score(y_test, test_scores)
test_pr_auc = average_precision_score(y_test, test_scores)

# confusion matrix and specificity
test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_preds).ravel()
test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0.0


# =========================
# SAVE TEST RESULTS
# =========================
# save test predictions and scores
test_results = X_test.copy()
test_results["anomaly_score"] = test_scores
test_results["predicted_anomaly"] = test_preds
test_results["anomaly_label"] = y_test

test_results.to_csv(RESULTS_DIR / "ocsvm_test_scores_predictions.csv", index=False)


# =========================
# SAVE METRICS
# =========================
# store all metrics for further comparison
metrics_df = pd.DataFrame([{
    "model": "OneClassSVM",
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