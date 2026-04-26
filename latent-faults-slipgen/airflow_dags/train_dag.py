from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'train_mapper_decoder_pipeline',
    default_args=default_args,
    description='Pipeline for training models via Docker',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    # The workspace is mounted to /app inside the Airflow container.
    # The Dockerfile already installed requirements.txt into the container's Python environment.
    
    tune_hyperparams_task = BashOperator(
        task_id='tune_hyperparameters',
        bash_command='cd /app && python scripts/tune_mapper.py',
    )

    train_model_task = BashOperator(
        task_id='train_model',
        bash_command='cd /app && dvc repro',
    )

    tune_hyperparams_task >> train_model_task
