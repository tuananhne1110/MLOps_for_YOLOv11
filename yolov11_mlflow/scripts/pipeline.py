#!/usr/bin/env python3
"""Main pipeline script for automating the entire training workflow."""
import os
import sys
import logging
import subprocess
import time
import requests
from pathlib import Path
import yaml
from datetime import datetime

# Get the root repository path (parent of yolov11_mlflow)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Get the yolov11_mlflow path
yolov11_root = Path(__file__).parent.parent

def setup_logging():
    """Set up logging configuration."""
    log_dir = yolov11_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd or str(project_root), 
                              capture_output=True, text=True)
        logging.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}")
        logging.error(f"Error: {e}")
        logging.error(f"Stderr: {e.stderr}")
        return False

def check_services() -> bool:
    """Check if MLflow and MinIO services are running."""
    logging.info("Checking MLflow and MinIO services...")
    
    def check_service(url: str, name: str) -> bool:
        try:
            response = requests.get(url, timeout=5)
            # For MinIO, 403 means the service is running but requires authentication
            if response.status_code in [200, 403]:
                logging.info(f"{name} service is running")
                return True
            else:
                logging.error(f"{name} service returned status code {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error checking {name} service: {e}")
            return False
    
    mlflow_ready = check_service("http://localhost:5000", "MLflow")
    minio_ready = check_service("http://localhost:9000", "MinIO")
    
    if not mlflow_ready or not minio_ready:
        logging.error("Required services (MLflow, MinIO) are not running")
        return False
    
    return True

def check_dvc_changes() -> bool:
    """Check if there are any changes in DVC tracked files."""
    logging.info("Checking for DVC changes...")
    
    # Pull latest changes from Git
    if not run_command("git pull"):
        logging.error("Failed to pull Git changes")
        return False
    
    # Check DVC status
    try:
        result = subprocess.run("dvc status", shell=True, capture_output=True, text=True)
        if result.returncode == 0 and not result.stdout.strip():
            logging.info("No DVC changes detected")
            return False
        logging.info("DVC changes detected")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking DVC status: {e}")
        return False

def pull_data() -> bool:
    """Pull latest data from DVC storage."""
    logging.info("Pulling data from DVC storage...")
    return run_command("dvc pull --force")

def train_model() -> bool:
    """Run model training."""
    logging.info("Starting model training...")
    
    # Get current timestamp for run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo11_run_{timestamp}"
    
    # Update training config with run name
    config_path = yolov11_root / "configs" / "training_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update MLflow run name
        config['mlflow']['run_name'] = run_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logging.error(f"Error updating training config: {e}")
        return False
    
    # Run training script
    train_script = yolov11_root / "src" / "training" / "train.py"
    return run_command(f"python {train_script.relative_to(project_root)} --config yolov11_mlflow/configs/training_config.yaml")

def export_model() -> bool:
    """Export model to engine format and upload to MinIO."""
    logging.info("Exporting and uploading model...")
    export_script = yolov11_root / "scripts" / "export_model.py"
    return run_command(f"python {export_script.relative_to(project_root)}")

def cleanup() -> bool:
    """Clean up temporary files and update Git."""
    logging.info("Cleaning up...")
    
    # Add and commit changes
    commands = [
        "git add .",
        'git commit -m "Auto update: Completed training pipeline"',
        "git push"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            logging.error(f"Failed to execute: {cmd}")
            return False
    
    return True

def run_pipeline():
    """Run the complete training pipeline."""
    setup_logging()
    logging.info("Starting pipeline...")
    
    try:
        # Check if services are running
        if not check_services():
            logging.error("Required services (MLflow, MinIO) are not running")
            return False
        
        # Check for DVC changes
        if not check_dvc_changes():
            logging.info("No changes detected. Pipeline completed.")
            return True
        
        # Pull data
        if not pull_data():
            logging.error("Failed to pull data")
            return False
        
        # Train model
        if not train_model():
            logging.error("Failed to train model")
            return False
        
        # Export model
        if not export_model():
            logging.error("Failed to export model")
            return False
        
        # Cleanup
        if not cleanup():
            logging.error("Failed to cleanup")
            return False
        
        logging.info("Pipeline completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1) 
