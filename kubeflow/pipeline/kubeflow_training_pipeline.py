from kfp.dsl import pipeline
from kubeflow.components.load_component import ingest_training_data
from kubeflow.components.featured_component import build_features
from kubeflow.components.train_component import train_model
from kubeflow.components.evaluate_component import evaluate_model


@pipeline(name="mmm-training-pipeline")
def training_pipeline():
    query = """
        SELECT *
        FROM MARKETING_ML.ANALYTICS.PROCESSED_MARKETING_DATA
    """

    ingest_op = ingest_training_data(
        query=query, 
    )

    feature_op = build_features(
        input_path=ingest_op.outputs['output_path']
    )

    train_op = train_model(
        input_path=feature_op.outputs['output_path'],
    )

    evaluate_model(
        test_path=train_op.outputs['test_path'],
        model_artifact=train_op.outputs['model_artifact'],
    )


if __name__ == "__main__":
    training_pipeline()