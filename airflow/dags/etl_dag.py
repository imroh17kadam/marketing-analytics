import sys
sys.path.append("/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.ingestion.extract import extract_data
from src.preprocess.transform import transform_data
from src.preprocess.load import load_data

default_args = {
    'owner': 'data_engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

RAW_PATH = "data/raw/synthetic_mmm_data.csv"
PROCESSED_PATH = "data/processed/processed_mmm.csv"

def extract(**context):
    df = extract_data(RAW_PATH)
    context["ti"].xcom_push(key="raw_df", value=df)

def transform(**context):
    df = context["ti"].xcom_pull(key="raw_df")
    df_clean = transform_data(df)
    context["ti"].xcom_push(key="clean_df", value=df_clean)

def load(**context):
    df = context["ti"].xcom_pull(key="clean_df")
    load_data(df, PROCESSED_PATH)

with DAG(
    dag_id="marketing_sales_etl_v1",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["etl", "marketing"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform,
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load,
    )

    extract_task >> transform_task >> load_task