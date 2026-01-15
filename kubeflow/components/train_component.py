from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from pathlib import Path
import json

# @component(base_image="python:3.10")
def train_model(
    input_path: str | Path,
    model_artifact: str | Path,
    test_path: str | Path,
    target: str = "sales",
    alpha: float = 1.0,
    test_size: float = 0.2
):
    """
    Train Ridge MMM model and output test dataset.
    """
    import pandas as pd
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from src.common.constants import features_mmm

    input_path = Path(input_path)
    model_artifact = Path(model_artifact)
    test_path = Path(test_path)

    model_artifact.parent.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    X = df[features_mmm]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"✅ Model fitting completed.")

    joblib.dump(model, model_artifact)
    print(f"✅ Model saved to {model_artifact}")

    X_test.to_csv(test_path / "X_test.csv", index=False)
    y_test.to_csv(test_path / "y_test.csv", index=False)
    
    print(f"✅ Test data saved to {test_path}")