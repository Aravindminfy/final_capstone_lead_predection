from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator  # Use DummyOperator if using <2.3
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
    'ml_pipeline_end_to_end',
    default_args=default_args,
    description='Run full ML pipeline: EDA, Preprocessing, Training, Dashboard',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'streamlit']
)

# === Step 0: Placeholder Extract Task ===
extract = BashOperator(
    task_id='extract_postgres_data',
    bash_command='echo "docker does not have access to local host pgadmin"',
    dag=dag
)

# === Step 1: EDA with Streamlit ===
streamlit_command = (
    'nohup streamlit run /opt/airflow/data/streamlit_dash.py '
    '> /opt/airflow/data/streamlit.log 2>&1 &'
)

run_streamlit = BashOperator(
    task_id='EDA_stramlit_dashboard',
    bash_command=streamlit_command,
    dag=dag
)

# === Step 2: Drift Detection ===
drift_check = BashOperator(
    task_id='drift_check',
    bash_command='python /opt/airflow/dags/data_drift_check.PY',
    dag=dag
)

# === Step 3: Branch based on drift flag ===
def check_drift_flag():
    drift_flag_path = '/opt/airflow/data/drift_status.flag'
    if os.path.exists(drift_flag_path):
        with open(drift_flag_path, 'r') as f:
            status = f.read().strip().lower()
            if status == 'drift':
                return 'run_preprocess'
    return 'no_drift_task'

branch_drift = BranchPythonOperator(
    task_id='check_drift_branch',
    python_callable=check_drift_flag,
    dag=dag
)

# === Step 4: Pipeline Tasks ===
run_preprocess = BashOperator(
    task_id='run_preprocess',
    bash_command='python /opt/airflow/dags/preprocess.py',
    dag=dag
)

run_model_training = BashOperator(
    task_id='run_model_training',
    bash_command='python /opt/airflow/dags/modal.py',
    dag=dag
)

mlflow = BashOperator(
    task_id='mlflow_tracking',
    bash_command='echo "docker does not have access to local host mlflow"',
    dag=dag
)

# === Step 5: No Drift Task (empty Flask placeholder) ===
no_drift_task = EmptyOperator(
    task_id='no_drift_task',
    dag=dag
)

# === Task Dependencies ===
extract >> run_streamlit >> drift_check >> branch_drift

branch_drift >> run_preprocess >> run_model_training >> mlflow
branch_drift >> no_drift_task
