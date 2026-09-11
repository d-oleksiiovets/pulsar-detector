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
from sklearn.model_selection import learning_curve, cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def compute_metrics(y_true, y_pred, y_score=None, model_name: str | None = None) -> dict[str, np.ndarray]:
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

def plot_roc_curve(y_true, y_score, model_name: str | None = None, ax: plt.Axes | None = None, color: str | None = None,)-> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plot_kwargs = {"color": color} if color is not None else {}
    ax.plot(fpr, tpr, label=f"{model_name or 'Model'} (AUC = {auc:.3f})", alpha=0.6, linewidth=2, **plot_kwargs)
    if not any(line.get_label() == "Random classifier" for line in ax.lines):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, color="#cbd5e1")
    return ax


def plot_pr_curve(y_true, y_score, model_name: str | None = None, ax: plt.Axes | None = None, color: str | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plot_kwargs = {"color": color} if color is not None else {}
    ax.plot(recall, precision, label=f"{model_name or 'Model'} (AP = {ap:.3f})", alpha=0.6, linewidth=2, **plot_kwargs)
    baseline = float(np.mean(y_true))
    if not any(line.get_label().startswith("Baseline") for line in ax.lines):
        ax.axhline(y=baseline, linestyle="--", linewidth=1, color="#64748b", label=f"Baseline (AP = {baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2, color="#cbd5e1")
    return ax

def plot_confusion_matrix(y_true, y_pred, model_name: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-pulsar", "Pulsar"],
    )

    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name or 'Model'}")
    return ax

def plot_embeddings(
    projections: dict[str, np.ndarray],
    target: np.ndarray | pd.Series,
    target_names: dict[int, str] | list[str] | None = None,
    n_cols: int = 2,
    cmap: str = "coolwarm",
    alpha: float = 0.4,
    s: int = 8,
    is_categorical: bool = True,
    figsize_per_plot: tuple[int, int] = (6, 5)
):
    n_plots = len(projections)
    n_cols = min(n_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    last_scatter = None

    for ax, (name, X_proj) in zip(axes_flat, projections.items()):
        last_scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=target, cmap=cmap, alpha=alpha, s=s, edgecolors="none")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    if is_categorical:
        handles, raw_labels = last_scatter.legend_elements()
        if target_names is not None:
            if isinstance(target_names, dict):
                unique_vals = sorted(np.unique(np.asarray(target)))
                labels = [str(target_names.get(v, v)) for v in unique_vals]
            else:
                labels = target_names
        else:
            labels = raw_labels
        fig.legend(handles, labels, title="Class", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    else:
        cbar = fig.colorbar(last_scatter, ax=axes_flat[:n_plots], shrink=0.8, pad=0.02)
        cbar.set_label("Target Value")

    fig.tight_layout()
    return fig, axes

def compute_learning_curve(
    estimator,
    X, 
    y, 
    cv, 
    scoring: str = "average_precision", 
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 6), 
    n_jobs: int = -1
) -> dict[str, np.ndarray]:
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=42,
    )

    return {
        "train_sizes": train_sizes_abs,
        "train_mean": np.mean(train_scores, axis=1),
        "train_std": np.std(train_scores, axis=1),
        "val_mean": np.mean(val_scores, axis=1),
        "val_std": np.std(val_scores, axis=1),
    }

def plot_learning_curves(
    artifacts: dict[str, dict],
    scoring_name: str = "Average Precision",
    n_cols: int = 2,
    figsize_per_plot: tuple[int, int] = (6, 5),
) -> tuple[plt.Figure, np.ndarray]:

    models_with_lc = {
        name: data["learning_curve"]
        for name, data in artifacts.items()
        if "learning_curve" in data
    }

    n_plots = len(models_with_lc)
    n_cols = min(n_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (run_name, lc) in zip(axes_flat, models_with_lc.items()):
        sizes = lc["train_sizes"]

        ax.plot(sizes, lc["train_mean"], "o-", color="#2563eb", label="Train", linewidth=1.8, markersize=4)
        ax.fill_between(
            sizes,
            lc["train_mean"] - lc["train_std"],
            lc["train_mean"] + lc["train_std"],
            alpha=0.15,
            color="#2563eb"
        )

        ax.plot(sizes, lc["val_mean"], "o-", color="#dc2626", label="CV Validation", linewidth=1.8, markersize=4)
        ax.fill_between(
            sizes,
            lc["val_mean"] - lc["val_std"],
            lc["val_mean"] + lc["val_std"],
            alpha=0.15,
            color="#dc2626",
        )

        ax.set_title(run_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training instances")
        ax.set_ylabel(scoring_name)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right")

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes