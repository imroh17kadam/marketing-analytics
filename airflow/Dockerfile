# Use official Airflow image
FROM apache/airflow:2.7.1-python3.11

# Switch to root to install Python dependencies
USER root

# Install system packages if needed
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

# Copy dags folder
COPY dags/ /opt/airflow/dags/