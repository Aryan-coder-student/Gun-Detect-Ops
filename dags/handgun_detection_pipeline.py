from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys



sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import load_config, download_dataset, train_model, save_model


def load_config_wrapper():
    config = load_config("../config.yaml")
    return config


def download_task(**kwargs):
    config = kwargs['ti'].xcom_pull(task_ids='load_config')
    dataset_path = download_dataset(config)
    kwargs['ti'].xcom_push(key='dataset_path', value=dataset_path)


def train_task(**kwargs):
    config = kwargs['ti'].xcom_pull(task_ids='load_config')
    dataset_path = kwargs['ti'].xcom_pull(task_ids='download_dataset', key='dataset_path')
    model = train_model(dataset_path, config)
    save_model(model, config)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'handgun_detection_pipeline',
    default_args=default_args,
    description='Handgun Detection MLOps Pipeline',
    schedule_interval=timedelta(days=1),  
    start_date=datetime(2025, 4, 6),
    catchup=False,
) as dag:

    #
    t0 = PythonOperator(
        task_id='load_config',
        python_callable=load_config_wrapper,
        provide_context=True,
    )

    
    t1 = PythonOperator(
        task_id='download_dataset',
        python_callable=download_task,
        provide_context=True,
    )

    
    t2 = PythonOperator(
        task_id='train_model',
        python_callable=train_task,
        provide_context=True,
    )

    
    t3 = BashOperator(
        task_id='push_to_dvc',
        bash_command='dvc add data models && dvc push',
    )

    
    t4 = BashOperator(
        task_id='build_docker',
        bash_command='docker build -t handgun-detection-api:latest .',
    )

    t5 = BashOperator(
        task_id='run_docker',
        bash_command='docker run -d -p 8000:8000 handgun-detection-api:latest',
    )

    t0 >> t1 >> t2 >> t3 >> t4 >> t5