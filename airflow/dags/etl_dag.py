import sys
sys.path.append("/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from airflow.deployment.extract import DataExtractor
from airflow.deployment.load_raw_to_snowflake import load_raw_to_snowflake
from airflow.deployment.transform import transform_data
from airflow.deployment.load import load_data

# Datadog monitoring
# from plugins.datadog_monitoring import track_task


default_args = {
    'owner': 'data_engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

RAW_PATH = "data/raw/synthetic_mmm_data.csv"


# -------------------------
# TASK 1 — Extract + Load Raw
# -------------------------

def extract_and_load_raw():

    # monitor = track_task("marketing.airflow.extract_load_raw")

    # monitor.start()

    try:

        df_raw = DataExtractor.extract()

        load_raw_to_snowflake(
            df_raw,
            source="synthetic_csv"
        )

        # monitor.success()

    except Exception as e:

        # monitor.fail()

        raise e

    finally:
        print("Succeeded")
        # monitor.duration()


# -------------------------
# TASK 2 — Transform + Load Processed
# -------------------------

def transform_and_load_processed():

    # monitor = track_task("marketing.airflow.transform_load_processed")

    # monitor.start()

    try:

        df_raw = DataExtractor.extract()

        df_processed = transform_data(df_raw)

        load_data(df_processed)

        # monitor.success()

    except Exception as e:

        # monitor.fail()

        raise e

    finally:
        print("Succeeded")
        # monitor.duration()


# -------------------------
# DAG Definition
# -------------------------

with DAG(
    dag_id="marketing_sales_etl_v1",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
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