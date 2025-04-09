#!/usr/bin/env python3
"""Script to export trained YOLOv11 model to TensorRT engine format."""
import os
import sys
import logging
import argparse
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from ultralytics import YOLO
from datetime import datetime
import shutil

# Get the root repository path (parent of yolov11_mlflow)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Get the yolov11_mlflow path
yolov11_root = Path(__file__).parent.parent

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def create_s3_bucket_if_not_exists(endpoint_url, bucket_name, region="us-east-1"):
    """Create S3 bucket if it doesn't exist."""
    try:
        s3_client = boto3.client('s3', endpoint_url=endpoint_url)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"S3 bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404' or error_code == 'NoSuchBucket':
                try:
                    if region == 'us-east-1':
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region}
                        )
                    logging.info(f"Created S3 bucket '{bucket_name}'")
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

def upload_to_minio(local_path: Path, s3_path: str, bucket_name: str):
    """Upload file to MinIO."""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url="http://localhost:9000",
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        
        s3_client.upload_file(str(local_path), bucket_name, s3_path)
        logging.info(f"Uploaded {local_path} to {s3_path}")
        return True
    except Exception as e:
        logging.error(f"Error uploading to MinIO: {e}")
        return False

def clean_local_files(model_dir: Path):
    """Clean up local model directory."""
    try:
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logging.info(f"Removed directory {model_dir}")
    except Exception as e:
        logging.error(f"Error cleaning local files: {e}")

def get_latest_model() -> tuple[Path, str]:
    """Find the latest trained model in saved_models directory."""
    saved_models_dir = yolov11_root / "saved_models"
    if not saved_models_dir.exists():
        raise FileNotFoundError("saved_models directory not found")
    
    # Get all model directories
    model_dirs = [d for d in saved_models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError("No model directories found in saved_models")
    
    # Sort directories by creation time (newest first)
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = model_dirs[0]
    
    # Get the best.pt file
    model_path = latest_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"best.pt not found in {latest_dir}")
    
    return model_path, latest_dir.name

def export_model(model_path: str = None):
    """Export model to TensorRT engine format and upload to MinIO."""
    try:
        # If no model path provided, find the latest model
        if model_path is None:
            model_path, model_name = get_latest_model()
            logging.info(f"Found latest model: {model_path}")
        else:
            model_path = Path(model_path)
            model_name = model_path.parent.name
        
        # Create a temporary directory for renamed files
        temp_dir = model_path.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Rename and copy files
        new_pt_path = temp_dir / f"{model_name}.pt"
        shutil.copy2(model_path, new_pt_path)
        
        # Load model
        model = YOLO(str(new_pt_path))
        
        # Export to TensorRT engine
        logging.info("Exporting model to TensorRT engine format...")
        engine_path = model.export(format='engine', half=True)
        
        # Rename engine file
        new_engine_path = temp_dir / f"{model_name}.engine"
        shutil.move(engine_path, new_engine_path)
        
        # Upload to MinIO
        s3_bucket = "mlflow"
        s3_model_path = f"{model_name}.engine"
        s3_pt_path = f"{model_name}.pt"
        
        if not create_s3_bucket_if_not_exists("http://localhost:9000", s3_bucket):
            raise Exception("Failed to create/verify MinIO bucket")
            
        # Upload engine file
        if not upload_to_minio(new_engine_path, s3_model_path, s3_bucket):
            raise Exception("Failed to upload engine file to MinIO")
            
        # Upload original model file
        if not upload_to_minio(new_pt_path, s3_pt_path, s3_bucket):
            raise Exception("Failed to upload model file to MinIO")
            
        # Clean up local files
        clean_local_files(model_path.parent)
        
        logging.info("Model export and upload completed successfully")
        
    except Exception as e:
        logging.error(f"Error in export process: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to TensorRT engine format")
    parser.add_argument("--model-path", type=str, help="Path to the trained model (optional, will use latest if not provided)")
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        export_model(args.model_path)
    except Exception as e:
        logging.error(f"Failed to export model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 