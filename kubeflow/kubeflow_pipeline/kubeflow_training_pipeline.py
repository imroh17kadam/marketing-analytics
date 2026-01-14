from kfp.dsl import pipeline
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from kubeflow.components.load_component import ingest_training_data
from kubeflow.components.featured_component import build_features
from kubeflow.components.train_component import train_model
from kubeflow.components.evaluate_component import evaluate_model

# @pipeline(name="local-ml-training-pipeline")
def training_pipeline():
    query = """
            SELECT *
            FROM MARKETING_ML.ANALYTICS.PROCESSED_MARKETING_DATA
            """
    processed_sales_data = PROJECT_ROOT / "artifacts" / "processed_data" / "processed_sales_data.csv"
    feature_engineered_data = PROJECT_ROOT / "artifacts" / "featured_data" / "feature_engineered_sales_data.csv"
    model_path = PROJECT_ROOT / "artifacts" / "model" / "ridge_mmm_model.pkl"
    evaluation_data = PROJECT_ROOT / "artifacts" / "evaluation_data"

    
    ingest_training_data(query=query, output_data=str(processed_sales_data))

    build_features(input_data=str(processed_sales_data), output_data=feature_engineered_data)

    train_model(input_data=str(feature_engineered_data), model_path=model_path, test_output=evaluation_data, target="sales")

    evaluate_model(model_path=model_path, test_output=evaluation_data)

    print("Done")

if __name__ == "__main__":
    training_pipeline()