from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os
import logging

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Machine Learning Training Pipeline',
    schedule_interval='0 0 * * *',  # Chạy lúc 00:00 mỗi ngày
    catchup=False
)

# Define tasks
checkout = BashOperator(
    task_id='checkout',
    bash_command='''
        if [ -d "/opt/airflow/workspace/gfas" ]; then
            cd /opt/airflow/workspace/gfas && \
            git fetch && \
            git reset --hard origin/main
        else
            git clone --depth 1 https://github.com/tuananhne1110/gfas.git /opt/airflow/workspace/gfas
        fi
    ''',
    dag=dag
)

setup_env = BashOperator(
    task_id='setup_environment',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        if [ ! -d "venv" ]; then
            python -m venv venv && \
            . venv/bin/activate && \
            pip install -r requirements.txt
        else
            . venv/bin/activate && \
            if [ requirements.txt -nt venv/.requirements_installed ]; then
                pip install -r requirements.txt && \
                touch venv/.requirements_installed
            fi
        fi
    ''',
    dag=dag
)

configure_git = BashOperator(
    task_id='configure_git',
    bash_command='''
        if ! git config --global user.name > /dev/null; then
            git config --global user.name "Airflow"
        fi
        if ! git config --global user.email > /dev/null; then
            git config --global user.email "airflow@example.com"
        fi
    ''',
    dag=dag
)

configure_dvc = BashOperator(
    task_id='configure_dvc',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        # Initialize DVC if not already initialized
        if [ ! -d .dvc ]; then
            dvc init --no-scm
        fi
        # Configure DVC remote if not already configured
        if [ ! -f .dvc/.credentials_configured ] || [ ! -f .dvc/config.local ]; then
            dvc remote modify --local minio access_key_id {{ var.value.MINIO_ACCESS_KEY }} && \
            dvc remote modify --local minio secret_access_key {{ var.value.MINIO_SECRET_KEY }} && \
            touch .dvc/.credentials_configured
        fi
    ''',
    dag=dag
)

start_mlflow = BashOperator(
    task_id='start_mlflow',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        # Copy .env file if it doesn't exist
        if [ ! -f .env ]; then
            cp yolov11_mlflow/.env .env
        fi
        # Copy .env file to yolov11_mlflow directory
        cp .env yolov11_mlflow/.env
        # Start services
        python yolov11_mlflow/scripts/start_services.py
    ''',
    dag=dag
)

wait_for_mlflow = BashOperator(
    task_id='wait_for_mlflow',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        python -c "
import time
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mlflow():
    try:
        response = requests.get('http://localhost:5000')
        return response.status_code == 200
    except:
        return False

def check_minio():
    try:
        response = requests.get('http://localhost:9000')
        return response.status_code == 200
    except:
        return False

max_retries = 30
retry_interval = 5

for i in range(max_retries):
    logger.info(f'Checking MLflow and MinIO services (attempt {i+1}/{max_retries})...')
    if check_mlflow() and check_minio():
        logger.info('MLflow and MinIO services are ready!')
        exit(0)
    time.sleep(retry_interval)

logger.error('Services did not become ready in time')
exit(1)
"
    ''',
    dag=dag
)

pull_data = BashOperator(
    task_id='pull_data',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        # Check if data has changed on remote
        if dvc status | grep -q "changed"; then
            echo "DVC changes detected, pulling new data..."
            dvc pull --force
        else
            echo "No DVC changes detected, skipping training..."
            exit 0  # Exit with success to stop the pipeline
        fi
    ''',
    dag=dag
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='''
        cd /opt/airflow/workspace/gfas && \
        . venv/bin/activate && \
        # Set environment variables for MLflow and MinIO
        export MLFLOW_TRACKING_URI=http://localhost:5000 && \
        export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 && \
        export AWS_ACCESS_KEY_ID=minioadmin && \
        export AWS_SECRET_ACCESS_KEY=minioadmin && \
        python yolov11_mlflow/scripts/pipeline.py
    ''',
    dag=dag
)

# Define task dependencies
checkout >> setup_env >> configure_git >> configure_dvc >> start_mlflow >> wait_for_mlflow >> pull_data >> train_model 
