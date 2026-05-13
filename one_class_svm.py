from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import OneClassSVM
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
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "train_data.csv")
X_val = pd.read_csv(DATA_DIR / "validation_data.csv")

y_val = pd.read_csv(DATA_DIR / "validation_labels.csv")
y_val = y_val["anomaly_label"].astype(int).values

model = OneClassSVM(
    kernel="rbf",
    gamma=0.1,
    nu=0.2
)

model.fit(X_train)

joblib.dump(model, MODELS_DIR / "ocsvm.pkl")

val_scores = -model.decision_function(X_val)

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

val_results.to_csv(RESULTS_DIR / "ocsvm_validation_scores.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(val_scores, bins=50)
plt.axvline(best_threshold, linestyle="--")
plt.title("One-Class SVM validation anomaly score distribution")
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ocsvm_validation_scores_hist.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve)
plt.title("One-Class SVM validation PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ocsvm_validation_pr_curve.png", dpi=300)
plt.close()

metrics_df = pd.DataFrame([{
    "model": "OneClassSVM",
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
