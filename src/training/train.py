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
        # Resolve to local path
        original_path = dataset_config["path"]
        local_path = resolve_dvc_path(original_path)
        
        if str(local_path) != original_path:
            # Update the configuration to use local path
            dataset_config["path"] = str(local_path)
            
            # Create a temporary dataset configuration file
            temp_data_yaml = Path(local_path) / "dataset_resolved.yaml"
            with open(temp_data_yaml, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            logging.info(f"Resolved dataset path {original_path} to {local_path}")
            
            # Register cleanup function to run at exit
            atexit.register(cleanup_dvc)
            
            return str(temp_data_yaml)
    
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


def update_model_version(best_model_path: Path, model_name: str, results: Any) -> None:
    """Update the model version after successful training.
    
    Args:
        best_model_path: Path to the best model file
        model_name: Name of the model (e.g., 'yolov11n')
        results: Training results object containing metrics
    """
    try:
        # Get the models directory
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Create a backup of the old model if it exists
        old_model_path = models_dir / f"{model_name}.pt"
        if old_model_path.exists():
            backup_path = models_dir / f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy2(old_model_path, backup_path)
            logging.info(f"Created backup of old model at: {backup_path}")
        
        # Copy the new best model to replace the old one
        shutil.copy2(best_model_path, old_model_path)
        logging.info(f"Updated model at: {old_model_path}")
        
        # Update the model version in a version file
        version_file = models_dir / "model_version.txt"
        version_info = {
            "model_name": model_name,
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "path": str(old_model_path),
            "metrics": {
                "mAP50": results.results_dict["metrics/mAP50(B)"],
                "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
                "precision": results.results_dict["metrics/precision(B)"],
                "recall": results.results_dict["metrics/recall(B)"],
                "fitness": results.fitness
            }
        }
        
        with open(version_file, "w") as f:
            yaml.dump(version_info, f, default_flow_style=False)
        logging.info(f"Updated model version info at: {version_file}")
        
    except Exception as e:
        logging.error(f"Error updating model version: {e}")
        raise


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
        model_path = project_root / "models" / f"yolov11{model_name[-1]}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
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
            "cache": False,  # Force YOLO to regenerate cache files
            "save": checkpoint_config["save_best"],  # Whether to save best model
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
        }
        
        # Add save_period only if it's enabled
        if checkpoint_config["save_period"] > 0:
            args["save_period"] = checkpoint_config["save_period"]
        
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
                
            except Exception as e:
                logging.warning(f"Failed to log best model to MLflow: {e}")
                # Fallback: Copy model to local directory
                local_model_dir = project_root / "saved_models" / mlflow_logger.run_name
                local_model_dir.mkdir(parents=True, exist_ok=True)
                import shutil
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