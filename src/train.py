from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.data import load_data, save_model

def build_final_pipeline() -> Pipeline:
    """Build the final pipeline used to train the production model."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

def main():
    """Train the final model on the full dataset and save it to disk."""
    
    df = load_data()
    X, y = df.drop(columns="target"), df["target"]

    pipeline = build_final_pipeline()
    pipeline.fit(X, y)

    saved_path = save_model(pipeline)
    print(f"Model trained on {len(X)} samples and saved to {saved_path}")

if __name__ == "__main__":
    main()