import sys
sys.path.append("/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.ingestion.extract import extract_data
from src.preprocess.transform import transform_data

from src.ingestion.extract import extract_data
from src.preprocess.load_raw_to_snowflake import load_raw_to_snowflake
from src.preprocess.transform import transform_data
from src.preprocess.load import load_data

default_args = {
    'owner': 'data_engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

RAW_PATH = "data/raw/synthetic_mmm_data.csv"

def extract_and_load_raw():
    df_raw = extract_data(RAW_PATH)
    load_raw_to_snowflake(df_raw, source="synthetic_csv")

def transform_and_load_processed():
    # For now transform uses the extracted dataframe logic
    # In STEP 6 we will read directly from Snowflake
    df_raw = extract_data(RAW_PATH)
    df_processed = transform_data(df_raw)
    load_data(df_processed)

with DAG(
    dag_id="marketing_sales_etl_v2",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["etl", "marketing", "snowflake"],
) as dag:

    load_raw_task = PythonOperator(
        task_id="extract_and_load_raw",
        python_callable=extract_and_load_raw,
    )

    load_processed_task = PythonOperator(
        task_id="transform_and_load_processed",
        python_callable=transform_and_load_processed,
    )

    load_raw_task >> load_processed_task