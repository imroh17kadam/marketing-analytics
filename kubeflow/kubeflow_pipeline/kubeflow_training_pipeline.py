from kfp.dsl import pipeline
from kubeflow.components.load_component import ingest_training_data
from kubeflow.components.featured_component import build_features
from kubeflow.components.train_component import train_model
from kubeflow.components.evaluate_component import evaluate_model


# @pipeline(name="mmm-training-pipeline")
def training_pipeline():
    query = """
        SELECT *
        FROM MARKETING_ML.ANALYTICS.PROCESSED_MARKETING_DATA
    """

    ingest_op = ingest_training_data(
        query=query, 
        output_path="artifacts/ingestion/ingested_data.csv"
    )

    feature_op = build_features(
        input_path="artifacts/ingestion/ingested_data.csv",
        output_path="artifacts/features/feature_engineered_data.csv"
    )

    train_op = train_model(
        input_path="artifacts/features/feature_engineered_data.csv",
        model_artifact="artifacts/model/ridge_mmm_v1.pkl",
        test_path="artifacts/test/"
    )

    evaluate_model(
        input_path="artifacts/test/",
        model_artifact="artifacts/model/ridge_mmm_v1.pkl",
        evaluation_path="artifacts/evaluation/"
    )


if __name__ == "__main__":
    training_pipeline()