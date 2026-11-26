from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 2000
amount = rng.uniform(100, 10000, n)
volatility = rng.uniform(0, 1, n)
frequency = rng.integers(1, 50, n)
X = np.column_stack([amount, volatility, frequency])
score = ((amount > 7000).astype(int)
         + (volatility > 0.65).astype(int)
         + (frequency > 35).astype(int))
y = (score >= 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(
    n_estimators=150, max_depth=8,
    min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
accuracy = float(accuracy_score(y_test, model.predict(X_test)))
joblib.dump(model, ARTIFACTS / "risk_model.joblib")
metrics = {"dataset": "synthetic", "accuracy": accuracy,
           "training_samples": len(X_train),
           "testing_samples": len(X_test),
           "model": "RandomForestClassifier", "random_state": 42}
(ARTIFACTS / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8")
print("MLOps Model Training Completed")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Model accuracy: {accuracy:.2%}")
print(f"Model saved to: {ARTIFACTS}")