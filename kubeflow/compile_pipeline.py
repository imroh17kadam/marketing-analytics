from kfp import compiler
from kubeflow.pipeline.kubeflow_training_pipeline import training_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="mmm_training_pipeline.yaml"
    )