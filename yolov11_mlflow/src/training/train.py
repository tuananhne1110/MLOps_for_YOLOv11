"""Main training script for YOLOv11 with MLflow tracking."""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import atexit
from datetime import datetime
import shutil

import mlflow
import boto3
from botocore.exceptions import ClientError
from ultralytics import YOLO, settings

settings.update({"mlflow": False})

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_yaml_config, get_device, get_mlflow_config
from src.utils.mlflow_logger import MLflowLogger
from src.utils.dvc_fs import resolve_dvc_path, cleanup_dvc


def setup_logging(log_file: Optional[str] = None) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def prepare_dataset_config(data_yaml_path: str) -> str:
    """Prepare dataset configuration with DVC support.
    
    Args:
        data_yaml_path: Path to dataset YAML configuration
        
    Returns:
        Path to processed dataset configuration
    """
    # Read the original dataset configuration
    with open(data_yaml_path, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    # Resolve the path if needed
    if "path" in dataset_config and isinstance(dataset_config["path"], str):
        # Get the original path from config
        original_path = dataset_config["path"]
        
        # Get the datasets directory path
        datasets_dir = project_root / "data" / "datasets"
        temp_data_yaml = datasets_dir / "dataset.yaml"
        
        # Use DVC to pull data
        try:
            # Run DVC pull to get the data
            import subprocess
            result = subprocess.run(['dvc', 'pull', '--force'], capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"DVC pull failed: {result.stderr}")
                raise RuntimeError("Failed to pull data with DVC")
            
            logging.info("Successfully pulled data with DVC")
            
            # Remove existing dataset.yaml if it exists
            if temp_data_yaml.exists():
                temp_data_yaml.unlink()
                logging.info(f"Removed existing dataset.yaml at {temp_data_yaml}")
            
            # Remove all cache files in all subdirectories
            for cache_file in datasets_dir.rglob("*.cache"):
                try:
                    cache_file.unlink()
                    logging.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to remove cache file {cache_file}: {e}")
                    
            # Also check and remove cache files in train and valid directories
            for subdir in ['train', 'valid']:
                subdir_path = datasets_dir / subdir
                if subdir_path.exists():
                    for cache_file in subdir_path.glob("*.cache"):
                        try:
                            cache_file.unlink()
                            logging.info(f"Removed cache file from {subdir}: {cache_file}")
                        except Exception as e:
                            logging.warning(f"Failed to remove cache file {cache_file}: {e}")
            
            # Verify dataset structure
            train_dir = datasets_dir / "train"
            valid_dir = datasets_dir / "valid"
            
            if not (train_dir / "images").exists() or not (valid_dir / "images").exists():
                raise FileNotFoundError("Dataset structure is incorrect. Missing train/images or valid/images directories")
            
            # Update the configuration to use datasets path
            dataset_config["path"] = str(datasets_dir)
            dataset_config["train"] = "train/images"
            dataset_config["val"] = "valid/images"
            
            # Create a temporary dataset configuration file
            with open(temp_data_yaml, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            logging.info(f"Dataset configuration updated to use path: {datasets_dir}")
            
            return str(temp_data_yaml)
            
        except Exception as e:
            logging.error(f"Error preparing dataset: {e}")
            raise
    
    return data_yaml_path


def create_s3_bucket_if_not_exists(endpoint_url, bucket_name, region="us-east-1"):
    """Create S3 bucket if it doesn't exist.
    
    Args:
        endpoint_url: S3 endpoint URL
        bucket_name: S3 bucket name
        region: AWS region
        
    Returns:
        True if bucket exists or was created successfully, False otherwise
    """
    try:
        s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"S3 bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404' or error_code == 'NoSuchBucket':
                # Bucket doesn't exist, create it
                try:
                    if region == 'us-east-1':
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region}
                        )
                    logging.info(f"Created S3 bucket '{bucket_name}'")
                    
                    # Wait for bucket to be ready
                    waiter = s3_client.get_waiter('bucket_exists')
                    waiter.wait(Bucket=bucket_name)
                    logging.info(f"Bucket '{bucket_name}' is ready")
                    
                    return True
                except Exception as create_e:
                    logging.error(f"Failed to create S3 bucket: {create_e}")
                    return False
            else:
                logging.error(f"Error checking S3 bucket: {e}")
                return False
    except Exception as e:
        logging.error(f"Unexpected error with S3: {e}")
        return False


def setup_mlflow_local_storage():
    """Configure MLflow to use local storage."""
    # Create local mlruns directory
    local_artifact_path = project_root / "mlruns"
    local_artifact_path.mkdir(exist_ok=True)
    
    # Set MLflow to use local storage
    mlflow.set_tracking_uri(f"file:{local_artifact_path}")
    
    # Remove S3 environment variables
    for env_var in ["MLFLOW_S3_ENDPOINT_URL", "MLFLOW_S3_BUCKET_NAME", "MLFLOW_DEFAULT_ARTIFACT_ROOT"]:
        if env_var in os.environ:
            os.environ.pop(env_var)
    
    logging.info(f"MLflow configured to use local storage at {local_artifact_path}")


def get_model_versions() -> dict:
    """Get all model versions from version file.
    
    Returns:
        Dictionary containing all version information
    """
    try:
        version_file = project_root / "models" / "model_version.txt"
        if not version_file.exists():
            logging.warning("No version file found")
            return {}
            
        with open(version_file, "r") as f:
            version_info = yaml.safe_load(f)
        return version_info
    except Exception as e:
        logging.error(f"Error reading version file: {e}")
        return {}


def download_model_from_minio(s3_path: str, local_path: Path) -> bool:
    """Download model from MinIO.
    
    Args:
        s3_path: S3 path in format s3://bucket/path
        local_path: Local path to save model
        
    Returns:
        bool: True if successful
    """
    try:
        # Parse S3 path
        s3_path = s3_path.replace("s3://", "")
        bucket = s3_path.split("/")[0]
        key = "/".join(s3_path.split("/")[1:])
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url="http://localhost:9000",
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        
        # Download file
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logging.info(f"Downloaded model from {s3_path} to {local_path}")
        return True
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        return False


def rollback_model_version(version: str) -> bool:
    """Rollback to a specific model version.
    
    Args:
        version: Version timestamp to rollback to
        
    Returns:
        bool: True if successful
    """
    try:
        # Get version info
        version_info = get_model_versions()
        if not version_info or version_info["version"] != version:
            logging.error(f"Version {version} not found")
            return False
        
        # Get S3 path
        s3_path = version_info["s3_path"]
        model_name = version_info["model_name"]
        
        # Download model to models directory
        model_path = project_root / "models" / f"{model_name}.pt"
        if not download_model_from_minio(s3_path, model_path):
            return False
        
        logging.info(f"Successfully rolled back to version {version}")
        return True
    except Exception as e:
        logging.error(f"Error rolling back version: {e}")
        return False


def update_model_version(best_model_path: Path, model_name: str, results: Any) -> None:
    """Update the model version information.
    
    Args:
        best_model_path: Path to the best model file
        model_name: Name of the model (e.g., 'yolov11n')
        results: Training results object containing metrics
    """
    try:
        # Get the models directory
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Get current version info
        current_version = get_model_versions()
        
        # Create new version info
        new_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_info = {
            "model_name": model_name,
            "version": new_version,
            "s3_path": f"s3://{mlflow_bucket}/{model_name}/versions/{new_version}/model.pt",  # Include version in path
            "previous_version": current_version.get("version") if current_version else None,
            "metrics": {
                "mAP50": results.results_dict["metrics/mAP50(B)"],
                "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
                "precision": results.results_dict["metrics/precision(B)"],
                "recall": results.results_dict["metrics/recall(B)"],
                "fitness": results.fitness
            }
        }
        
        # Save version info
        version_file = models_dir / "model_version.txt"
        with open(version_file, "w") as f:
            yaml.dump(version_info, f, default_flow_style=False)
        logging.info(f"Updated model version info at: {version_file}")
        
        # Upload model to versioned path in MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url="http://localhost:9000",
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        
        s3_path = f"{model_name}/versions/{new_version}/model.pt"
        s3_client.upload_file(str(best_model_path), mlflow_bucket, s3_path)
        logging.info(f"Uploaded model to versioned path: {s3_path}")
        
        # Also update best.pt for latest version
        latest_path = f"{model_name}/best.pt"
        s3_client.upload_file(str(best_model_path), mlflow_bucket, latest_path)
        logging.info(f"Updated latest model at: {latest_path}")
        
    except Exception as e:
        logging.error(f"Error updating model version: {e}")
        raise


def download_model(model_name: str, model_path: Path) -> bool:
    """Download YOLOv11 model if it doesn't exist.
    
    Args:
        model_name: Name of the model (e.g., 'yolo11n')
        model_path: Path where the model should be saved
        
    Returns:
        bool: True if model was downloaded successfully
    """
    try:
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if model_path.exists():
            logging.info(f"Model already exists at {model_path}")
            return True
        
        logging.info(f"Downloading {model_name} model...")
        
        # Try to download using ultralytics
        try:
            # First try with yolo prefix
            model = YOLO(f"yolo/{model_name}")
            model.save(str(model_path))
            logging.info(f"Model downloaded and saved to {model_path}")
            return True
        except Exception as e1:
            logging.warning(f"Failed to download with yolo/ prefix: {e1}")
            try:
                # Try with direct name
                model = YOLO(model_name)
                model.save(str(model_path))
                logging.info(f"Model downloaded and saved to {model_path}")
                return True
            except Exception as e2:
                logging.error(f"Failed to download model directly: {e2}")
                # Try downloading from hub
                try:
                    from ultralytics import hub
                    model = hub.get_model(model_name)
                    model.save(str(model_path))
                    logging.info(f"Model downloaded from hub and saved to {model_path}")
                    return True
                except Exception as e3:
                    logging.error(f"Failed to download from hub: {e3}")
                    raise Exception(f"All download attempts failed for model {model_name}")
        
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return False


def train_yolov11(config_path: str) -> None:
    """Train YOLOv11 model using Ultralytics and track with MLflow.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = load_yaml_config(config_path)
    training_config = config["training"]
    dataset_config = config["dataset"]
    mlflow_config = config["mlflow"]
    checkpoint_config = config["checkpoint"]
    
    # Set up MLflow configuration
    mlflow_settings = get_mlflow_config()
    
    # Use separate buckets for data and model artifacts
    s3_endpoint = "http://localhost:9000"  # Match DVC S3 endpoint
    data_bucket = "dvc"  # Bucket for data
    global mlflow_bucket  # Make mlflow_bucket accessible in other functions
    mlflow_bucket = "mlflow"  # Bucket for model artifacts
    
    # Try to create both buckets if they don't exist
    data_bucket_exists = create_s3_bucket_if_not_exists(s3_endpoint, data_bucket)
    mlflow_bucket_exists = create_s3_bucket_if_not_exists(s3_endpoint, mlflow_bucket)
    
    # Setup MLflow storage
    if data_bucket_exists and mlflow_bucket_exists:
        logging.info(f"Using S3 storage for data: {data_bucket}")
        logging.info(f"Using S3 storage for MLflow artifacts: {mlflow_bucket}")
        
        # Set DVC bucket for data
        os.environ["DVC_S3_BUCKET"] = data_bucket
        
        # Set MLflow bucket for artifacts
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
        os.environ["MLFLOW_S3_BUCKET_NAME"] = mlflow_bucket
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"s3://{mlflow_bucket}/"
        os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{mlflow_bucket}/"
        
        logging.info(f"Data bucket: {data_bucket}")
        logging.info(f"MLflow artifact root: {os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT']}")
        logging.info(f"S3 endpoint: {s3_endpoint}")
        
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(mlflow_settings["tracking_uri"])
        mlflow.set_experiment(mlflow_config["experiment_name"])
        
        # Verify both buckets exist
        try:
            s3_client = boto3.client('s3', endpoint_url=s3_endpoint)
            s3_client.head_bucket(Bucket=data_bucket)
            s3_client.head_bucket(Bucket=mlflow_bucket)
            logging.info(f"Verified buckets exist: {data_bucket}, {mlflow_bucket}")
        except Exception as e:
            logging.error(f"Error verifying buckets: {e}")
            raise
    else:
        logging.info("Using local storage for MLflow artifacts")
        setup_mlflow_local_storage()
    
    # Set up MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name=mlflow_config["experiment_name"],
        run_name=mlflow_config["run_name"],
        tags=mlflow_config["tags"],
    )
    
    try:
        # Start MLflow run
        mlflow_logger.start_run()
        
        # Log parameters
        mlflow_logger.log_params({
            "training": training_config,
            "dataset": dataset_config,
            "checkpoint": checkpoint_config,
        })
        
        # Load model
        model_name = training_config["model_name"]
        device = get_device(training_config["device"])
        
        # Update model path to use models directory
        model_path = project_root / "models" / f"{model_name}.pt"
        logging.info(f"Model path: {model_path}")
        
        # Download model if it doesn't exist
        if not download_model(model_name, model_path):
            logging.error(f"Failed to download model {model_name}")
            logging.error("Available YOLO models: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x")
            raise FileNotFoundError(f"Failed to download model {model_name}")
        
        # Verify model exists after download
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path} after download attempt")
        
        logging.info(f"Loading model from {model_path}")
        model = YOLO(str(model_path))  # Load pretrained model
        
        # Prepare dataset configuration with DVC support
        data_yaml = prepare_dataset_config(dataset_config["data_yaml"])
        
        # Get training arguments
        args = {
            "data": data_yaml,
            "epochs": training_config["epochs"],
            "imgsz": training_config["img_size"],
            "batch": training_config["batch_size"],
            "device": device,
            "workers": training_config["workers"],
            "patience": training_config["patience"],
            "val": training_config["val"],  # Whether to perform validation
            "cache": "disk",  # Use disk cache instead of RAM
            "save": True,  # Save best model only
            "save_period": -1,  # Disable periodic saving
            "project": str(project_root / "runs"),
            "name": mlflow_logger.run_name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": training_config["optimizer"]["name"],
            "lr0": training_config["optimizer"]["lr"],
            "weight_decay": training_config["optimizer"]["weight_decay"],
            "hsv_h": training_config["augmentation"]["hsv_h"],
            "hsv_s": training_config["augmentation"]["hsv_s"],
            "hsv_v": training_config["augmentation"]["hsv_v"],
            "degrees": training_config["augmentation"]["degrees"],
            "translate": training_config["augmentation"]["translate"],
            "scale": training_config["augmentation"]["scale"],
            "shear": training_config["augmentation"]["shear"],
            "fliplr": training_config["augmentation"]["fliplr"],
            "mosaic": training_config["augmentation"]["mosaic"],
            "mixup": training_config["augmentation"]["mixup"],
            "rect": False,  # Disable rectangular training
            "resume": False,  # Disable resume from last checkpoint
        }
        
        # Log YOLOv11 hyperparameters to MLflow
        mlflow_logger.log_params({"yolo_args": args})
        
        # Start training
        logging.info(f"Starting YOLOv11 training with {model_name} model")
        results = model.train(**args)
        
        # Log metrics from training
        metrics = {
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
            "fitness": results.fitness,
        }
        mlflow_logger.log_metrics(metrics)
        
        # Log training plots
        for plot_path in [
            results.save_dir / "results.png",
            results.save_dir / "confusion_matrix.png",
            results.save_dir / "labels.jpg",
            results.save_dir / "val_batch0_pred.jpg",
        ]:
            if plot_path.exists():
                try:
                    mlflow_logger.log_artifact(plot_path)
                except Exception as e:
                    logging.warning(f"Failed to log artifact {plot_path}: {e}")
        
        # Log the best model to MLflow
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            try:
                # Log the model file as an artifact first
                mlflow_logger.log_artifact(best_model_path)
                
                # Then log the model with the correct parameters
                mlflow.pyfunc.log_model(
                    "best_model",
                    loader_module="src.models.mlflow_model_wrapper",
                    data_path=str(best_model_path),
                    registered_model_name="YOLOv11"
                )
                logging.info(f"Best model logged to MLflow: {best_model_path}")
                
                # Update the model version with results
                update_model_version(best_model_path, model_name, results)
                
                # Upload best model to MinIO
                s3_client = boto3.client(
                    's3',
                    endpoint_url=s3_endpoint,
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin'
                )
                
                # Create bucket if it doesn't exist
                try:
                    s3_client.head_bucket(Bucket=mlflow_bucket)
                except:
                    s3_client.create_bucket(Bucket=mlflow_bucket)
                
                # Upload best model to MinIO
                s3_path = f"{model_name}/best.pt"
                s3_client.upload_file(str(best_model_path), mlflow_bucket, s3_path)
                logging.info(f"Best model uploaded to MinIO: {s3_path}")
                
            except Exception as e:
                logging.warning(f"Failed to log best model to MLflow: {e}")
                # Fallback: Copy model to local directory
                local_model_dir = project_root / "saved_models" / mlflow_logger.run_name
                local_model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_model_path, local_model_dir)
                logging.info(f"Best model saved locally to {local_model_dir}")
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.exception(f"Error during training: {e}")
        raise
    
    finally:
        # End MLflow run
        mlflow_logger.end_run()
        
        # Clean up DVC temporary files
        cleanup_dvc()


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train YOLOv11 model with MLflow tracking")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--log-file", 
        type=str, 
        default="logs/training.log",
        help="Path to log file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set up logging
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    setup_logging(args.log_file)
    
    # Start training
    train_yolov11(args.config)
