from fastapi import FastAPI, HTTPException
from api.schemas import PulsarFeatures, PredictionResponse
from api.predictor import predict, load

app = FastAPI(
    title="Pulsar Star Detection API",
    description="Binary classification of HTRU2 pulsar candidates via Logistic Regression.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    load()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: PulsarFeatures, threshold: float = 0.5):
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    return predict(features, threshold=threshold)