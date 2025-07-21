from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import subprocess
import sys
import boto3
import json

# === Install All Required Packages ===


def install_all_packages():
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas", "boto3", "evidently", "numpy", "scikit-learn",
                           "joblib", "matplotlib", "seaborn", "sagemaker"])

# === Detect Drift and Push XCom ===


def detect_drift(**context):
    import pandas as pd
    import boto3
    import json
    from evidently import Report
    from evidently.presets import DataDriftPreset

    s3 = boto3.client('s3')
    bucket = 'scripts-aravind'
    key = 'script-airflow/drift.py'
    local_path = '/tmp/drift.py'
    s3.download_file(bucket, key, local_path)

    # Run script and capture result
    result = subprocess.run([sys.executable, local_path],
                            capture_output=True, text=True)

    # Logic to detect drift (e.g., script prints "Drift: True")
    drift_detected = "Drift: True" in result.stdout

    # Push XCom
    context['ti'].xcom_push(key='drift_detected', value=drift_detected)

# === Branch to Training if Drift ===


def decide_branch(**context):
    drift_detected = context['ti'].xcom_pull(
        key='drift_detected', task_ids='detect_drift')
    return 'train_model' if drift_detected else 'skip_training'

# === Run modalrun.py Script ===


def run_training_script():
    s3 = boto3.client('s3')
    bucket = 'scripts-aravind'
    key = 'script-airflow/modalrun.py'
    local_path = '/tmp/modalrun.py'
    s3.download_file(bucket, key, local_path)
    subprocess.run([sys.executable, local_path], check=True)


# === DAG Definition ===
with DAG(
    dag_id='drift_detection_and_conditional_training',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["drift", "conditional", "training"]
) as dag:

    install = PythonOperator(
        task_id='install_dependencies',
        python_callable=install_all_packages
    )

    detect = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        provide_context=True
    )

    branch = BranchPythonOperator(
        task_id='check_drift_branch',
        python_callable=decide_branch,
        provide_context=True
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=run_training_script
    )

    skip = EmptyOperator(task_id='skip_training')

    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # DAG Flow
    install >> detect >> branch
    branch >> train >> end
    branch >> skip >> end
