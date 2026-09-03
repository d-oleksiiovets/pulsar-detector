from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np

def compute_metrics(y_true, y_pred, y_score=None, model_name: str | None = None) -> dict:
    metrics = {
        "Model": model_name or "Model",
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": np.nan,
        "PR-AUC (AP)": np.nan,
    }

    if y_score is not None:
        y_score_arr = np.asarray(y_score)
        if y_score_arr.ndim == 2 and y_score_arr.shape[1] == 2:
            y_score_arr = y_score_arr[:, 1]

        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score_arr)
        metrics["PR-AUC (AP)"] = average_precision_score(y_true, y_score_arr)
    
    return metrics

def plot_roc_curve(y_true, y_score, model_name: str | None = None, ax: plt.Axes | None = None)-> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.plot(fpr, tpr, label=f"{model_name or 'Model'} (AUC = {auc:.3f})", linewidth=2, color="#1f77b4")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.2, color="#cbd5e1")
    return ax


def plot_pr_curve(y_true, y_score, model_name: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.plot(recall, precision, label=f"{model_name or 'Model'} (AP = {ap:.3f})", linewidth=2, color="#1f77b4")
    baseline = float(y_true.mean())
    plt.axhline(y=baseline, linestyle="--", linewidth=1, color="#64748b", label=f"Baseline (AP = {y_true.mean():.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.2, color="#cbd5e1")
    return ax

def plot_confusion_matrix(y_true, y_pred, model_name: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-pulsar", "Pulsar"],
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name or 'Model'}")
    return ax