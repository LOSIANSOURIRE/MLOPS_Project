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
    description='Pipeline for training mapper and decoder models',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    tune_hyperparams_task = BashOperator(
        task_id='tune_hyperparameters',
        bash_command='python /app/scripts/tune_mapper.py',
    )

    train_model_task = BashOperator(
        task_id='train_model',
        bash_command='cd /app && mlflow run . -P learning_rate=0.01 -P batch_size=32 --env-manager=local',
    )

    tune_hyperparams_task >> train_model_task
    
    train_model_task