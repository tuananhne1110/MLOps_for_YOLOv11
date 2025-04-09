"""Configuration utilities for YOLOv11 training."""
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_mlflow_config() -> Dict[str, str]:
    """Get MLflow configuration from environment variables.
    
    Returns:
        Dictionary containing MLflow configuration
    """
    return {
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "yolov11-training"),
        "s3_endpoint_url": os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        "s3_bucket_name": os.environ.get("MLFLOW_S3_BUCKET_NAME", "mlflow"),
    }


def get_run_name() -> str:
    """Generate a run name based on current timestamp.
    
    Returns:
        Run name string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"yolov11_run_{timestamp}"


def add_project_root_to_path() -> None:
    """Add project root directory to Python path."""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def get_device(device_id: Union[int, str]) -> str:
    """Get device string for torch.
    
    Args:
        device_id: Device ID (int) or 'cpu'
        
    Returns:
        Device string for torch
    """
    if device_id == "cpu":
        return "cpu"
    return f"cuda:{device_id}" if isinstance(device_id, int) else device_id 