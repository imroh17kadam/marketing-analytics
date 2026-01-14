from kfp.dsl import component, Output, Dataset
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


# @component(
#     base_image="python:3.10",
#     packages_to_install=["pandas", "snowflake-connector-python"]
# )
def ingest_training_data(
    query: str,
    output_data: str
):
    """
    Kubeflow component to ingest data using existing DataIngestion logic
    """

    from src.ingestion.ingestion import DataIngestion

    ingestion = DataIngestion(
        source="snowflake",
        query=query
    )

    df = ingestion.load()

    output_path = Path(output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(df)

    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    query = """
            SELECT *
            FROM MARKETING_ML.ANALYTICS.PROCESSED_MARKETING_DATA
            """
    output_file = PROJECT_ROOT / "artifacts" / "processed_data" / "processed_sales_data.csv"

    ingest_training_data(query=query, output_data=str(output_file))