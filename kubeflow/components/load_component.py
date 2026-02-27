
from kfp.dsl import Dataset, Output, component


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "snowflake-connector-python"],
)
def ingest_training_data(query: str, output_path: Output[Dataset]):
    """
    Ingest marketing data from Snowflake and output as a dataset artifact.
    """
    import pandas as pd

    from src.marketing_analytics.ingestion.ingestion import DataIngestion

    # output_path = Path(output_path)
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    ingestion = DataIngestion(source="snowflake", query=query)

    df: pd.DataFrame = ingestion.load()
    df.to_csv(output_path.path, index=False)

    print(f"✅ Data saved to {output_path.path}")
