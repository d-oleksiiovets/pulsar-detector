import pandas as pd
from api.schemas import PulsarFeatures
from src.data import load_model

_model = None

def load():
    """Load the model and cache it for subsequent predictions."""

    global _model
    if _model is None:
        _model = load_model()
    return _model

def  predict(features: PulsarFeatures, threshold: float = 0.5) -> dict:
    """Generate a pulsar prediction and probability from input features."""
    
    model = load()

    X = pd.DataFrame([features.model_dump()])
    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= threshold)

    return {
        "prediction": prediction,
        "probability": float(proba),
        "threshold_used": threshold,
    }