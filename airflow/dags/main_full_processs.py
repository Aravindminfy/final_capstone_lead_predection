from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
    'ml_pipeline_with_dashboard',
    default_args=default_args,
    description='Run full ML pipeline: EDA, Preprocessing, Training, Dashboard',
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=['ml', 'streamlit']
)
extract = BashOperator(
    task_id='extract_postgres_data',
    # bash_command='python /opt/airflow/dags/extract_pgadmin.py',  # Disabled: placeholder phase
    bash_command='echo "docker does not have access to local host pgadmin"',  # No-op command
    dag=dag
)

# Bash command to run Streamlit in background
streamlit_command = (
    'nohup streamlit run /opt/airflow/data/streamlit_dash.py '
    '> /opt/airflow/data/streamlit.log 2>&1 &'
)

run_streamlit = BashOperator(
    task_id='EDA_stramlit_dashboard',
    bash_command=streamlit_command,
    dag=dag
)


# STEP 2: Preprocessing
run_preprocess = BashOperator(
    task_id='run_preprocess',
    bash_command='python /opt/airflow/dags/preprocess.py',
    dag=dag
)

# STEP 3: Model Training
run_model_training = BashOperator(
    task_id='run_model_training',
    bash_command='python /opt/airflow/dags/modal.py',
    dag=dag
)
# STEP 4: MLflow Tracking
mlflow = BashOperator(
    task_id='mlflow_tracking',
    #bash_command='python /opt/airflow/dags/mlflow_tracking.py',
    bash_command='echo "docker does not have access to local host mlflow"',  # No-op command
    dag=dag
)

# step 5: Data Drift Check
drift_check = BashOperator(
    task_id='drift_check',
    bash_command='python /opt/airflow/dags/data_drift_check.PY',
    dag=dag
)

# Define task dependencies
extract >> run_streamlit >> run_preprocess >> run_model_training >> mlflow 
drift_check