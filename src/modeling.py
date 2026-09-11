from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
import umap

from src.features import add_coefficient_of_variation

def build_baseline_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", random_state=42))
    ])

def build_pipeline(model, use_smote=False) -> ImbPipeline:
    steps = [("feature_engineering", FunctionTransformer(add_coefficient_of_variation))]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("Model", model))
    return ImbPipeline(steps)

def build_linear_pipeline(model, use_smote=False):
    steps = [
        ("feature_engineering", FunctionTransformer(add_coefficient_of_variation)),
        ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
    ]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("Model", model))
    return ImbPipeline(steps)

def build_pca_pipeline(n_components: int = 2) -> Pipeline:
    return Pipeline([
        ("scaler", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("model", PCA(n_components=n_components, random_state=42))
    ])

def build_umap_pipeline(n_components: int = 2) -> Pipeline:
    return Pipeline([
        ("scaler", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("model", umap.UMAP(n_components=n_components, random_state=42))
    ])