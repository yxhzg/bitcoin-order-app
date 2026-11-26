import json
from pathlib import Path
import joblib

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

def test_model_and_metrics():
    model = joblib.load(ARTIFACTS / "risk_model.joblib")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    assert hasattr(model, "predict")
    assert metrics["dataset"] == "synthetic"
    assert metrics["accuracy"] >= 0.80
    assert metrics["training_samples"] == 1500
    assert metrics["testing_samples"] == 500