"""Utility functions for MLflow logging."""
import os
import mlflow
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from src.utils.config import get_mlflow_config, get_run_name


class MLflowLogger:
    """MLflow experiment tracker for YOLOv11 training."""
    
    def __init__(
        self, 
        experiment_name: Optional[str] = None, 
        run_name: Optional[str] = None, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name of MLflow run
            tags: Dictionary of tags to add to run
        """
        mlflow_config = get_mlflow_config()
        
        self.tracking_uri = mlflow_config["tracking_uri"]
        self.experiment_name = experiment_name or mlflow_config["experiment_name"]
        self.run_name = run_name or get_run_name()
        self.tags = tags or {}
        
        # Set S3 configuration
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_config["s3_endpoint_url"]
        # Always use "dvc" bucket for consistency
        os.environ["MLFLOW_S3_BUCKET_NAME"] = "dvc"
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "s3://dvc/"
        os.environ["MLFLOW_ARTIFACT_ROOT"] = "s3://dvc/"
        
        # Connect to MLflow tracking server
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)
        
        self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        self.run_id = None
        self.active = False
        
        logging.info(f"MLflow logger initialized: {self.experiment_name}/{self.run_name}")
        logging.info("Using S3 bucket: dvc")
        logging.info("Artifact root: s3://dvc/")
        logging.info(f"Tracking URI: {self.tracking_uri}")
    
    def start_run(self) -> None:
        """Start a new MLflow run."""
        if self.active:
            logging.warning("MLflow run already active, stopping current run first")
            self.end_run()
        
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=self.run_name)
        self.run_id = run.info.run_id
        self.active = True
        
        # Set tags
        mlflow.set_tags(self.tags)
        
        # Set artifact root for this run
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        logging.info(f"Started MLflow run: {self.run_id}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active:
            mlflow.end_run()
            self.active = False
            logging.info(f"Ended MLflow run: {self.run_id}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.active:
            logging.warning("MLflow run not active, starting new run")
            self.start_run()
        
        # Convert nested dict to flat dict with dot notation
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (optional)
        """
        if not self.active:
            logging.warning("MLflow run not active, starting new run")
            self.start_run()
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact to MLflow.
        
        Args:
            artifact_path: Path to artifact file
        """
        if not self.active:
            logging.warning("MLflow run not active, starting new run")
            self.start_run()
        
        # Ensure we're using the correct artifact root
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        mlflow.log_artifact(str(artifact_path))
    
    def log_model(self, model_path: Union[str, Path], name: str = "model") -> None:
        """Log YOLOv11 model to MLflow model registry.
        
        Args:
            model_path: Path to model file or directory
            name: Name for the logged model
        """
        if not self.active:
            logging.warning("MLflow run not active, starting new run")
            self.start_run()
        
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path=name,
            code_path=["src"],  # Include source code
            artifacts={"model_path": str(model_path)},
            loader_module="src.models.mlflow_model_wrapper",
            python_model=None,  # Will be loaded from loader_module
        )
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested dictionaries
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items) 