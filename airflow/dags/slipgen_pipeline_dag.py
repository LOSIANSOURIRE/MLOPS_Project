import os
import yaml
import random
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.exceptions import AirflowFailException

PROJECT_ROOT = "/opt/airflow/project/latent-faults-slipgen"
DEPLOY_MODELS_DIR = "/opt/airflow/project/deploy/backend/models"

def check_drift_func(**kwargs):
    config_path = "/opt/airflow/project/config.yaml"
    
    # Read config file to get the drift threshold
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Default to 0.15 if not found in config
            drift_threshold = config.get("data_drift_threshold", 0.15) 
    except FileNotFoundError:
        drift_threshold = 0.15
        print(f"Config not found at {config_path}, using default threshold: {drift_threshold}")

    # Dummy drift calculation (random simulated drift)
    current_drift = random.uniform(0.0, 0.3)
    print(f"Calculated drift: {current_drift:.3f} | Threshold: {drift_threshold}")

    if current_drift > drift_threshold:
        print("ALERT: Data drift detected! Triggering retraining pipeline.")
        return 'trigger_retraining'
    else:
        print("Data drift is within acceptable range.")
        return 'skip_retraining'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'slipgen_retraining_pipeline',
    default_args=default_args,
    description='A DAG to check data drift and trigger the slipgen model retraining',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'slipgen'],
) as dag:

    # Task 1: Check data drift and branch
    check_data_drift = BranchPythonOperator(
        task_id='check_data_drift',
        python_callable=check_drift_func,
    )

    # Task 2: Trigger model retraining
    # NOTE: Path modified to simply run the Python file given in the prompt, inside its virtual environment
    trigger_retraining = BashOperator(
        task_id='trigger_retraining',
        bash_command=f"cd {PROJECT_ROOT} && source .venv/bin/activate && python train_mapper_decoder.py",
    )

    # Task 3: Deploy the new model
    deploy_model = BashOperator(
        task_id='deploy_model',
        bash_command=f"mkdir -p {DEPLOY_MODELS_DIR} && cp {PROJECT_ROOT}/models/*.pth {DEPLOY_MODELS_DIR}/",
    )

    # Empty task to join the branch if retraining is skipped
    skip_retraining = EmptyOperator(
        task_id='skip_retraining'
    )

    # Set up task dependencies
    check_data_drift >> trigger_retraining >> deploy_model
    check_data_drift >> skip_retraining
