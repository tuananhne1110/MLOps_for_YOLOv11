from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os
import time
import requests
import logging
import subprocess

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
    schedule_interval=None,
    catchup=False
)

def wait_for_mlflow():
    """Wait for MLflow server to be ready."""
    # Use Docker network to access MLflow
    mlflow_uri = "http://localhost:5000"  # Use localhost since we're port-forwarding
    max_retries = 30
    retry_interval = 5

    logging.info("Waiting for MLflow server to be ready...")
    
    # First check if the containers are running
    try:
        # Check MLflow container
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mlflow-server", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        if "Up" not in result.stdout:
            logging.error("MLflow container is not running!")
            logging.error("Docker ps output: %s", result.stdout)
            # Check MLflow logs
            logs = subprocess.run(
                ["docker", "logs", "mlflow-server"],
                capture_output=True,
                text=True
            )
            logging.error("MLflow container logs: %s", logs.stdout)
            raise Exception("MLflow container is not running")

        # Check MinIO container
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mlflow-minio", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        if "Up" not in result.stdout:
            logging.error("MinIO container is not running!")
            logging.error("Docker ps output: %s", result.stdout)
            # Check MinIO logs
            logs = subprocess.run(
                ["docker", "logs", "mlflow-minio"],
                capture_output=True,
                text=True
            )
            logging.error("MinIO container logs: %s", logs.stdout)
            raise Exception("MinIO container is not running")

        # Check Postgres container
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mlflow-postgres", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        if "Up" not in result.stdout:
            logging.error("Postgres container is not running!")
            logging.error("Docker ps output: %s", result.stdout)
            # Check Postgres logs
            logs = subprocess.run(
                ["docker", "logs", "mlflow-postgres"],
                capture_output=True,
                text=True
            )
            logging.error("Postgres container logs: %s", logs.stdout)
            raise Exception("Postgres container is not running")

        # Wait for MLflow server to be ready
        for i in range(max_retries):
            try:
                response = requests.get(mlflow_uri)
                if response.status_code == 200:
                    logging.info("MLflow server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                logging.info(f"MLflow server not ready yet, retrying in {retry_interval} seconds... (attempt {i+1}/{max_retries})")
                time.sleep(retry_interval)
        
        raise Exception(f"MLflow server did not become ready after {max_retries} attempts")
        
    except Exception as e:
        logging.error(f"Error waiting for MLflow: {e}")
        raise

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
        # Create and setup virtual environment
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
        fi && \
        # Create .env file with necessary configurations
        echo "# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# MLflow Experiment Settings
MLFLOW_EXPERIMENT_NAME=yolov11-training-new
MLFLOW_S3_BUCKET_NAME=mlflow

# DVC Settings
DVC_REMOTE_NAME=dvc
DVC_REMOTE_URL=s3://dvc
DVC_S3_ENDPOINT_URL=http://localhost:9000

# Database settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow
POSTGRES_HOST=localhost
POSTGRES_PORT=5432" > yolov11_mlflow/.env && \
        cp yolov11_mlflow/.env .env
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
        # Add and configure minio remote
        dvc remote list | grep -q "minio" || dvc remote add -d minio s3://dvc
        dvc remote modify minio endpointurl http://localhost:9000
        dvc remote modify --local minio access_key_id {{ var.value.MINIO_ACCESS_KEY }}
        dvc remote modify --local minio secret_access_key {{ var.value.MINIO_SECRET_KEY }}
        touch .dvc/.credentials_configured
    ''',
    dag=dag
)

start_mlflow = BashOperator(
    task_id='start_mlflow',
    bash_command='cd /opt/airflow/workspace/gfas && python yolov11_mlflow/scripts/start_services.py',
    dag=dag
)

wait_for_mlflow = PythonOperator(
    task_id='wait_for_mlflow',
    python_callable=wait_for_mlflow,
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

# Set task dependencies
checkout >> setup_env >> configure_dvc >> start_mlflow >> wait_for_mlflow >> pull_data >> train_model 