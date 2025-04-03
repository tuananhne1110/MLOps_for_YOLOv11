#!/usr/bin/env python
"""Script to validate dataset structure and format."""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import yaml
import cv2

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def validate_directory_structure(dataset_path: str) -> bool:
    """Validate dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        bool: True if structure is valid
    """
    required_dirs = [
        "train/images",
        "train/labels",
        "test/images",
        "test/labels",
        "valid/images",
        "valid/labels"
    ]
    
    for dir_path in required_dirs:
        full_path = Path(dataset_path) / dir_path
        if not full_path.exists():
            logging.error(f"Missing required directory: {dir_path}")
            return False
    
    return True

def validate_dataset_yaml(dataset_path: str) -> Dict:
    """Validate dataset.yaml file.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dict: Dataset configuration if valid
    """
    yaml_path = Path(dataset_path) / "dataset.yaml"
    if not yaml_path.exists():
        logging.error("Missing dataset.yaml file")
        return None
    
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        required_fields = ["path", "train", "val", "test", "names"]
        for field in required_fields:
            if field not in config:
                logging.error(f"Missing required field in dataset.yaml: {field}")
                return None
        
        # Update paths to match new structure
        config["train"] = "train/images"
        config["val"] = "valid/images"
        config["test"] = "test/images"
        
        return config
    except Exception as e:
        logging.error(f"Error reading dataset.yaml: {e}")
        return None

def validate_labels(dataset_path: str, config: Dict) -> bool:
    """Validate label files and format.
    
    Args:
        dataset_path: Path to dataset directory
        config: Dataset configuration
        
    Returns:
        bool: True if labels are valid
    """
    valid = True
    class_ids = set(config["names"].keys())
    
    for split in ["train", "test", "valid"]:
        label_dir = Path(dataset_path) / split / "labels"
        image_dir = Path(dataset_path) / split / "images"
        
        # Get all label files
        label_files = list(label_dir.glob("*.txt"))
        image_files = list(image_dir.glob("*.jpg"))
        
        # Check if label files match image files
        label_names = {f.stem for f in label_files}
        image_names = {f.stem for f in image_files}
        
        if label_names != image_names:
            logging.error(f"Mismatch between images and labels in {split} split")
            logging.error(f"Images without labels: {image_names - label_names}")
            logging.error(f"Labels without images: {label_names - image_names}")
            valid = False
        
        # Validate label format
        for label_file in label_files:
            try:
                with open(label_file) as f:
                    for line_num, line in enumerate(f, 1):
                        values = line.strip().split()
                        if len(values) != 5:
                            logging.error(f"Invalid label format in {label_file}:{line_num}")
                            valid = False
                            continue
                        
                        class_id = int(values[0])
                        if class_id not in class_ids:
                            logging.error(f"Invalid class ID {class_id} in {label_file}:{line_num}")
                            valid = False
                        
                        # Validate normalized coordinates
                        for value in values[1:]:
                            try:
                                value = float(value)
                                if not 0 <= value <= 1:
                                    logging.error(f"Invalid coordinate value {value} in {label_file}:{line_num}")
                                    valid = False
                            except ValueError:
                                logging.error(f"Invalid number format in {label_file}:{line_num}")
                                valid = False
            except Exception as e:
                logging.error(f"Error reading label file {label_file}: {e}")
                valid = False
    
    return valid

def validate_images(dataset_path: str) -> bool:
    """Validate image files.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        bool: True if images are valid
    """
    valid = True
    
    for split in ["train", "test", "valid"]:
        image_dir = Path(dataset_path) / split / "images"
        
        for image_file in image_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(image_file))
                if img is None:
                    logging.error(f"Invalid image file: {image_file}")
                    valid = False
            except Exception as e:
                logging.error(f"Error reading image {image_file}: {e}")
                valid = False
    
    return valid

def validate_dataset(dataset_path: str) -> bool:
    """Validate entire dataset structure and content.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        bool: True if dataset is valid
    """
    logging.info(f"Validating dataset at: {dataset_path}")
    
    # Check directory structure
    if not validate_directory_structure(dataset_path):
        return False
    
    # Check dataset.yaml
    config = validate_dataset_yaml(dataset_path)
    if not config:
        return False
    
    # Check labels
    if not validate_labels(dataset_path, config):
        return False
    
    # Check images
    if not validate_images(dataset_path):
        return False
    
    logging.info("Dataset validation completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate dataset structure and format")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        if validate_dataset(args.dataset_path):
            logging.info("Dataset is valid and ready for use")
        else:
            logging.error("Dataset validation failed")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 