from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import sys
import boto3

# === STEP 1: Install Required Packages ===


def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas", "boto3", "evidently"])

# === STEP 2: Add Imports and Run drift.py ===


def run_drift_script():
    import pandas as pd
    import boto3
    import json
    from datetime import datetime
    from evidently import Report
    from evidently.presets import DataDriftPreset

    # Download script from S3
    s3 = boto3.client('s3')
    bucket = 'scripts-aravind'
    key = 'script-airflow/drift.py'
    local_path = '/tmp/drift.py'
    s3.download_file(bucket, key, local_path)

    # Execute the downloaded script
    subprocess.run([sys.executable, local_path], check=True)


# === DAG Definition ===
with DAG(
    dag_id='run_drift_analysis_from_s3',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["evidently", "data-drift"]
) as dag:

    t1 = PythonOperator(
        task_id='install_dependencies',
        python_callable=install_packages
    )

    t2 = PythonOperator(
        task_id='run_drift_script',
        python_callable=run_drift_script
    )

    t1 >> t2
