from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import sys
import boto3
import os

# === STEP 1: Install Required Python Libraries ===


def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas", "numpy", "scikit-learn", "joblib", "boto3",
                           "matplotlib", "seaborn", "sagemaker"])

# === STEP 2: Download and Run Script from S3 ===


def run_s3_script():
    s3 = boto3.client('s3')
    bucket = 'scripts-aravind'
    key = 'script-airflow/modalrun.py'
    local_path = '/tmp/modalrun.py'

    # Download script
    s3.download_file(bucket, key, local_path)

    # Execute script
    subprocess.run([sys.executable, local_path], check=True)


# === DAG Definition ===
with DAG(
    dag_id='install_and_run_modalrun_script',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["runtime", "s3", "ml"]
) as dag:

    t1 = PythonOperator(
        task_id='install_dependencies',
        python_callable=install_packages
    )

    t2 = PythonOperator(
        task_id='run_modalrun_from_s3',
        python_callable=run_s3_script
    )

    t1 >> t2
