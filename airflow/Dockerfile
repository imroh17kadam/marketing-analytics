# Use official Airflow image
FROM apache/airflow:2.8.1-python3.11

ENV AIRFLOW_HOME=/opt/airflow

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Copy DAGs – useful for prod, not required for local
COPY dags/ /opt/airflow/dags/