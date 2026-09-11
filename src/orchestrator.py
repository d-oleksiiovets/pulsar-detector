from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd
import optuna
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline

from src.optuna_search import run_search
from src.evaluate import compute_metrics, compute_learning_curve
from src.data import save_data

@dataclass
class ModelConfig:
    name: str
    model_fn: Callable[[dict[str, Any]], Any]
    param_space_fn: Callable[[optuna.Trial, bool], dict[str, Any]]
    pipeline_fn: Callable[[Any, bool], ImbPipeline]

def run_benchmark(
    experiment_name: str,
    models: list[ModelConfig],
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    cv: Any,
    smote_options: list[bool] = [False, True],
    n_trials: int = 50,
    scoring: str = "average_precision",
    seed: int = 42,
    save_df: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:

    artifacts: dict[str, dict[str, Any]] = {}
    summary_records: list[dict[str, Any]] = []
    summary_metrics: list[dict[str, np.ndarray]] = []

    for model_cfg in models:
        for use_smote in smote_options:
            smote_suffix = "smote" if use_smote else "raw"
            run_name = f"{model_cfg.name}__{smote_suffix}"
            model_name = f"{model_cfg.name} (Optuna, {smote_suffix})"

            study, best_pipeline = run_search(
                experiment_name=experiment_name,
                run_name=run_name,
                model_fn=model_cfg.model_fn,
                model_type=model_cfg.name,
                param_space_fn=model_cfg.param_space_fn,
                pipeline_fn=model_cfg.pipeline_fn,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                n_trials=n_trials,
                use_smote=use_smote,
                scoring=scoring,
                seed=seed,
            )

            y_pred = best_pipeline.predict(X_test)
            y_probs = best_pipeline.predict_proba(X_test)[:, 1]

            summary_metrics.append(
                compute_metrics(y_true=y_test, y_pred=y_pred, y_score=y_probs, model_name=model_name)
            )

            lc_data = compute_learning_curve(estimator=best_pipeline, X=X_train, y=y_train, cv=cv, scoring=scoring)

            artifacts[run_name] = {
                "study": study,
                "best_pipeline": best_pipeline,
                "best_params": study.best_params,
                "y_score": y_probs,
                "y_pred": y_pred,
                "learning_curve": lc_data,
            }

            summary_records.append(
                {
                    "model": model_cfg.name,
                    "use_smote": use_smote,
                    f"cv_{scoring}_mean": study.best_value,
                    "cv_std": study.best_trial.user_attrs.get("cv_std", None),
                    "best_params": study.best_params,
                }
            )

    summary_records_df = pd.DataFrame(summary_records).sort_values(
        by=f"cv_{scoring}_mean", ascending=False).reset_index(drop=True)

    summary_metrics_df = pd.DataFrame(summary_metrics).sort_values(
        by="PR-AUC (AP)", ascending=False).round(4)

    if save_df:
        save_data(summary_records_df, "tables/summary_records.csv")
        save_data(summary_metrics_df, "tables/summary_metrics.csv")
    
    return summary_metrics_df, summary_records_df, artifacts