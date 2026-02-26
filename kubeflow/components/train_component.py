from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
        "mlflow"
    ]
)
def train_model(
    input_path: Input[Dataset],
    model_artifact: Output[Model],
    test_path: Output[Dataset],
    target: str = "sales",
    alpha: float = 1.0,
    test_size: float = 0.2
):
    import pandas as pd
    import joblib
    import mlflow
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from src.common.constants import features_mmm

    # Set MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Start MLflow run
    with mlflow.start_run(run_name="kubeflow_ridge_training") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("target", target)

        df = pd.read_csv(input_path.path)

        X = df[features_mmm]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        print(f"✅ Model fitting completed.")

        # Save locally for Kubeflow artifact
        joblib.dump(model, model_artifact.path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="ridge_model",
            registered_model_name="MMM-Ridge-Model"
        )

        # Save test data
        X_test.to_csv(test_path.path / "X_test.csv", index=False)
        y_test.to_csv(test_path.path / "y_test.csv", index=False)

        print(f"✅ Model logged & registered in MLflow")