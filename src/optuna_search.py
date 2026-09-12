from typing import Callable, Any
import numpy as np
import mlflow
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score

def make_objective(
    model_fn: Callable[[dict[str, Any]], Any],
    param_space_fn: Callable[[optuna.Trial], dict[str, Any]],
    pipeline_fn: Callable[[optuna.Trial, bool], dict[str, Any]],
    X_train: Any,
    y_train: Any,
    cv: Any,
    use_smote: bool = False,
    scoring: str = "average_precision",
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for model hyperparameter optimization."""

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial, use_smote)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_param("use_smote", use_smote)

            pipeline = pipeline_fn(model_fn(params), use_smote=use_smote)

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(cv_scores))
            std_score = float(np.std(cv_scores))

            mlflow.log_metric("cv_mean_score", mean_score)
            mlflow.log_metric("cv_std_score", std_score)

            trial.set_user_attr("cv_std", std_score)
            return mean_score
    return objective

def run_search(
    experiment_name: str,
    run_name: str,
    model_fn: Callable[[dict[str, Any]], Any],
    model_type: str,
    param_space_fn: Callable[[optuna.Trial, bool], dict[str, Any]],
    X_train: Any,
    y_train: Any,
    cv: Any,
    pipeline_fn: Callable[[Any, bool], ImbPipeline],
    n_trials: int = 50,
    use_smote: bool = False,
    scoring: str = "average_precision",
    seed: int = 42,
) -> tuple[optuna.Study, Any]:
    """Run an Optuna hyperparameter search with MLflow tracking."""

    mlflow.set_experiment(experiment_name)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tags({
            "optimizer": "optuna",
            "scoring": scoring,
            "use_smote": str(use_smote),
            "model_type": model_type
        })

        objective = make_objective(
            model_fn=model_fn,
            param_space_fn=param_space_fn,
            pipeline_fn=pipeline_fn,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            use_smote=use_smote,
            scoring=scoring,
        )
        study.optimize(objective, n_trials=n_trials, catch=(Exception,), show_progress_bar=False)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_mean_score", study.best_value)

        best_pipeline = pipeline_fn(model_fn(study.best_params), use_smote=use_smote)
        best_pipeline.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="best_model",
            serialization_format="cloudpickle",
            input_example=X_train[:5] if hasattr(X_train, "__getitem__") else None
        )

    cv_std = study.best_trial.user_attrs.get("cv_std", 0.0)
    print(f"✓ {run_name} finished | Best CV {scoring}: {study.best_value:.4f} (±{cv_std:.4f})")

    return study, best_pipeline