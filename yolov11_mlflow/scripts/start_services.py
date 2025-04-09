#!/usr/bin/env python
"""Script to start MLflow and related services."""
import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path
import requests

# Get the root repository path (parent of yolov11_mlflow)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Get the yolov11_mlflow path
yolov11_root = Path(__file__).parent.parent

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = yolov11_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")
    # Try to load from project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")


def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Docker installed: {result.stdout.decode().strip()}")
        
        result = subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Docker daemon is running")
        return True
    except subprocess.CalledProcessError:
        logger.error("Docker is not running or not installed properly")
        return False
    except FileNotFoundError:
        logger.error("Docker command not found. Please install Docker")
        return False


def get_docker_compose_version():
    """Get the version of docker-compose."""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        version_str = result.stdout.decode().strip()
        logger.info(f"Docker Compose: {version_str}")
        return version_str
    except Exception:
        return "unknown"


def load_env_file():
    """Load environment variables from .env file."""
    env_file = yolov11_root / ".env"
    if env_file.exists():
        logger.info("Loading environment variables from .env file")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    else:
        logger.warning(".env file not found")


def check_services_running():
    """Check if MLflow services are already running."""
    services = ["mlflow-server", "mlflow-minio", "mlflow-postgres"]
    running_services = []
    
    for service in services:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={service}", "--format", "{{.Status}}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if "Up" in result.stdout.decode().strip():
                running_services.append(service)
        except subprocess.CalledProcessError:
            continue
    
    return running_services


def check_minio_buckets():
    """Check and create MinIO buckets if they don't exist."""
    logger.info("Checking MinIO buckets...")
    
    # Get MinIO credentials
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    
    # Buckets to check/create
    buckets = ["mlflow", "dvc"]
    
    for bucket in buckets:
        try:
            # Check if bucket exists
            result = subprocess.run(
                ["mc", "ls", f"myminio/{bucket}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"Bucket {bucket} exists")
        except subprocess.CalledProcessError:
            # Create bucket if it doesn't exist
            logger.info(f"Creating bucket {bucket}...")
            try:
                subprocess.run(
                    ["mc", "mb", f"myminio/{bucket}"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Set bucket policy to public
                subprocess.run(
                    ["mc", "policy", "set", "download", f"myminio/{bucket}"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info(f"Bucket {bucket} created and policy set")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create bucket {bucket}: {e}")
                logger.error(f"STDERR: {e.stderr.decode()}")
                return False
    
    return True


def start_services():
    """Start MLflow and related services using Docker Compose."""
    if not check_docker():
        logger.error("Cannot start services without Docker")
        sys.exit(1)
    
    # Load environment variables first
    load_env_file()
    
    # Check if services are already running
    running_services = check_services_running()
    required_services = ["mlflow-server", "mlflow-minio", "mlflow-postgres"]
    
    # If all services are running, we're good
    if all(service in running_services for service in required_services):
        logger.info("All required services are already running")
        # Check buckets anyway
        check_minio_buckets()
        return
    
    # If some services are running but not all, stop them first
    if running_services:
        logger.warning(f"Some services are running but not all: {', '.join(running_services)}")
        logger.info("Stopping existing services to ensure clean state...")
        stop_services()
    
    logger.info("Starting MLflow and related services...")
    
    try:
        # Start services using docker-compose
        compose_file = yolov11_root / "docker-compose.yml"
        
        if not compose_file.exists():
            logger.error(f"Docker Compose file not found: {compose_file}")
            sys.exit(1)
        
        # Get docker-compose version
        docker_compose_version = get_docker_compose_version()
        
        # Setup the command based on the docker-compose version
        cmd = ["docker-compose", "-f", str(compose_file), "up", "-d"]
        
        # Create environment variables by reading from .env
        env = os.environ.copy()
        env_file = yolov11_root / ".env"
        
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        env[key] = value
        
        # Ensure MLflow uses the same bucket as DVC
        dvc_remote_name = env.get("DVC_REMOTE_NAME", "dvc")
        env["MLFLOW_S3_BUCKET_NAME"] = dvc_remote_name
        
        # Run docker-compose
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        
        logger.info("Services started successfully")
        logger.info(result.stdout.decode())
        
        # Wait for services to be ready
        wait_for_services()
        
        # Check and create MinIO buckets
        check_minio_buckets()
        
        # Initialize DVC repository
        initialize_dvc()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start services: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        sys.exit(1)
    

def wait_for_services():
    """Wait for all services to be ready."""
    logger.info("Waiting for services to be ready...")
    
    # MLflow endpoint
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # MinIO endpoint
    minio_uri = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    
    # Wait for MLflow - the server needs time to initialize
    service_ready = False
    retries = 0
    max_retries = 30
    
    logger.info("Waiting for MLflow to initialize (this may take a minute)...")
    # Initial sleep to allow MLflow time to start up
    time.sleep(15)  
    
    while not service_ready and retries < max_retries:
        try:
            # Check if containers are running
            containers_running = check_services_running()
            if len(containers_running) < 3:
                logger.warning(f"Not all containers are running. Running: {containers_running}")
                time.sleep(5)
                retries += 1
                continue
            
            # Check MLflow UI
            ui_response = requests.get(mlflow_uri)
            if ui_response.status_code == 200:
                logger.info("MLflow UI is accessible")
                
                # Check MinIO
                try:
                    minio_response = requests.get(minio_uri)
                    if minio_response.status_code in [200, 403]:  # 403 means auth required, which is fine
                        logger.info("MinIO is accessible")
                        service_ready = True
                    else:
                        logger.warning(f"MinIO returned status code: {minio_response.status_code}")
                except requests.exceptions.ConnectionError:
                    logger.warning("MinIO not ready yet")
                
            else:
                logger.info(f"MLflow not ready yet (status code: {ui_response.status_code}), retrying...")
            
            if not service_ready:
                time.sleep(5)
                retries += 1
                
        except requests.exceptions.ConnectionError:
            logger.info("MLflow not ready yet, retrying...")
            time.sleep(5)
            retries += 1
    
    if not service_ready:
        logger.error("Services did not become ready in the expected time")
        logger.error("Check container logs with: docker logs -f mlflow-server")
        logger.error("Check container logs with: docker logs -f mlflow-minio")
        logger.error("Check container logs with: docker logs -f mlflow-postgres")
        sys.exit(1)
    else:
        logger.info("All services are ready and accessible")


def initialize_dvc():
    """Initialize DVC repository with default configuration."""
    logger.info("Initializing DVC repository...")
    
    # Check if DVC is installed
    try:
        result = subprocess.run(
            ["dvc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"DVC installed: {result.stdout.decode().strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("DVC is not installed or not in PATH. Please install DVC to use dataset versioning.")
        logger.warning("You can install it with: pip install dvc dvc-s3")
        return
    
    try:
        # Create directories if they don't exist
        dvc_repo_dir = yolov11_root / "data" / "dvc_repo"
        dvc_repo_dir.mkdir(parents=True, exist_ok=True)
        
        datasets_dir = yolov11_root / "data" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to the repo directory
        current_dir = os.getcwd()
        os.chdir(str(dvc_repo_dir))
        
        # Initialize DVC with --no-scm flag
        if not (dvc_repo_dir / ".dvc").exists():
            subprocess.run(["dvc", "init", "--no-scm"], check=True)
            
            # Configure DVC remote
            remote_name = os.getenv("DVC_REMOTE_NAME", "dvc")
            remote_url = os.getenv("DVC_REMOTE_URL", "s3://dvc")
            s3_endpoint = os.getenv("DVC_S3_ENDPOINT_URL", "http://localhost:9000")
            
            subprocess.run(["dvc", "remote", "add", "--default", remote_name, remote_url], check=True)
            subprocess.run(["dvc", "remote", "modify", remote_name, "endpointurl", s3_endpoint], check=True)
            
            logger.info("DVC repository initialized successfully")
        
        # Change back to original directory
        os.chdir(current_dir)
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize DVC repository: {e}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr.decode()}")
        else:
            logger.error("No error output available")


def show_service_info():
    """Show information about the running services."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    minio_console = "http://localhost:9001"
    
    logger.info("\n" + "=" * 60)
    logger.info("Services Information:")
    logger.info("-" * 60)
    logger.info(f"MLflow UI:       {mlflow_uri}")
    logger.info(f"MinIO Console:   {minio_console}")
    logger.info("-" * 60)
    logger.info("MinIO Credentials (used by MLflow & DVC):")
    logger.info(f"   Access Key:   {os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')}")
    logger.info(f"   Secret Key:   {os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')}")
    logger.info("-" * 60)
    logger.info("MLflow Settings:")
    logger.info(f"   S3 Bucket:    {os.getenv('MLFLOW_S3_BUCKET_NAME', 'dvc')}")
    logger.info(f"   S3 Endpoint:  {os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')}")
    logger.info("-" * 60)
    logger.info("DVC Settings:")
    logger.info(f"   Remote Name:  {os.getenv('DVC_REMOTE_NAME', 'dvc')}")
    logger.info(f"   Remote URL:   {os.getenv('DVC_REMOTE_URL', 's3://dvc')}")
    logger.info(f"   S3 Endpoint:  {os.getenv('DVC_S3_ENDPOINT_URL', 'http://localhost:9000')}")
    logger.info("=" * 60)


def stop_services():
    """Stop all running services."""
    logger.info("Stopping MLflow and related services...")
    
    try:
        # Stop services using docker-compose
        compose_file = yolov11_root / "docker-compose.yml"
        
        if not compose_file.exists():
            logger.error(f"Docker Compose file not found: {compose_file}")
            sys.exit(1)
        
        # Create environment variables by reading from .env
        env = os.environ.copy()
        env_file = yolov11_root / ".env"
        
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        env[key] = value
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "down"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        
        logger.info("Services stopped successfully")
        logger.info(result.stdout.decode())
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop services: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start or stop MLflow and related services")
    parser.add_argument(
        "--stop", 
        action="store_true",
        help="Stop the services instead of starting them"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.stop:
        stop_services()
    else:
        start_services()
        show_service_info() 
