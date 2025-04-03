#!/usr/bin/env python
"""Script to manage dataset versions and updates."""
import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import subprocess
from typing import List, Dict, Optional
from datetime import datetime

import mlflow
from dvc.api import DVCFileSystem
from dvc.repo import Repo

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def run_command(cmd: str, cwd: Optional[str] = None) -> bool:
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command '{cmd}': {e.stderr}")
        return False

def check_required_tools():
    """Check if required tools are installed."""
    required_tools = {
        "mc": "MinIO Client",
        "dvc": "DVC",
        "git": "Git"
    }
    
    missing_tools = []
    for tool, description in required_tools.items():
        if not shutil.which(tool):
            missing_tools.append(f"{description} ({tool})")
    
    if missing_tools:
        logging.error("Missing required tools:")
        for tool in missing_tools:
            logging.error(f"  - {tool}")
        
        logging.info("\nInstallation instructions:")
        logging.info("1. Install MinIO Client (mc):")
        logging.info("   wget https://dl.min.io/client/mc/release/linux-amd64/mc")
        logging.info("   chmod +x mc")
        logging.info("   sudo mv mc /usr/local/bin/")
        
        logging.info("\n2. Install DVC:")
        logging.info("   pip install dvc[s3]")
        
        logging.info("\n3. Install Git:")
        logging.info("   sudo apt-get update")
        logging.info("   sudo apt-get install git")
        
        return False
    return True

def setup_storage():
    """Set up MinIO buckets and DVC configuration."""
    logging.info("Setting up storage...")
    
    # Check for required tools
    if not check_required_tools():
        return False
    
    # Set up MinIO client alias
    if not run_command("mc alias set myminio http://localhost:9000 minioadmin minioadmin"):
        logging.error("Failed to set up MinIO alias")
        return False
    
    # Create MinIO bucket if it doesn't exist
    if not run_command("mc mb myminio/dvc --ignore-existing"):
        logging.error("Failed to create MinIO bucket")
        return False
    
    # Initialize Git if not already initialized
    if not (project_root / ".git").exists():
        logging.info("Initializing Git repository...")
        if not run_command("git init"):
            logging.error("Failed to initialize Git repository")
            return False
    
    # Initialize DVC if not already initialized
    if not (project_root / ".dvc").exists():
        if not run_command("dvc init --no-scm"):
            logging.error("Failed to initialize DVC")
            return False
    
    # Configure DVC remote with force flag to handle existing remote
    commands = [
        "dvc remote add -d -f minio s3://dvc",  # Added -f flag to force overwrite
        "dvc remote modify minio endpointurl http://localhost:9000",
        "dvc remote modify minio access_key_id minioadmin",
        "dvc remote modify minio secret_access_key minioadmin"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            logging.error(f"Failed to run command: {cmd}")
            return False
    
    return True

def get_dataset_info(dataset_path: str) -> Dict:
    """Get information about the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_info = {
        "classes": [],
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
    }
    
    # Read dataset.yaml
    yaml_path = Path(dataset_path) / "dataset.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
            dataset_info["classes"] = list(yaml_data.get("names", {}).values())
    
    # Count images
    for split in ["train", "val", "test"]:
        img_dir = Path(dataset_path) / split / "images"
        if img_dir.exists():
            dataset_info[f"{split}_images"] = len(list(img_dir.glob("*.jpg")))
    
    return dataset_info

def create_new_dataset(dataset_path: str, dataset_name: str, output_version: Optional[str] = None) -> bool:
    """Create a new dataset version.
    
    Args:
        dataset_path: Path to datasets directory
        dataset_name: Name of the dataset
        output_version: Optional version name for the new dataset
        
    Returns:
        bool: True if successful
    """
    logging.info(f"Creating new dataset at: {dataset_path}")
    
    # Set up storage first
    if not setup_storage():
        logging.error("Failed to set up storage")
        return False
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created dataset directory: {dataset_dir}")
    
    # Create subdirectories for images and labels
    for split in ["train", "valid", "test"]:
        # Create image directory
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        # Create labels directory
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create labels.cache for train split
        if split == "train":
            (dataset_dir / split / "labels.cache").touch(exist_ok=True)
    
    # Create dataset.yaml in the parent directory
    yaml_content = f"""
path: {dataset_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (relative to 'path')

# Classes
names:
  0: butter_sugar_bread
  1: chicken_floss_bread
  2: chicken_floss_sandwich
  3: cream_puf
  4: croissant
  5: donut
  6: muffin
  7: salted_egg_sponge_cake
  8: sandwich
  9: sponge_cake
  10: tiramisu
"""
    
    with open(dataset_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    # Add to DVC
    logging.info("Adding dataset to DVC...")
    if not run_command(f"dvc add {dataset_dir}"):
        return False
    
    # Add all files to Git
    logging.info("Adding files to Git...")
    if not run_command("git add ."):
        return False
    
    # Commit changes
    commit_msg = f"Add new dataset version {output_version if output_version else 'v1.0'}"
    if not run_command(f'git commit -m "{commit_msg}"'):
        return False
    
    # Push to MinIO
    logging.info("Pushing to MinIO...")
    if not run_command("dvc push"):
        return False
    
    logging.info(f"New dataset created successfully at: {dataset_dir}")
    return True

def update_dataset(
    dataset_path: str,
    dataset_name: str,
    current_version: str,
    new_products: Optional[List[str]] = None,
    new_data_path: Optional[str] = None,
    output_version: Optional[str] = None,
    remote_url: Optional[str] = None
) -> bool:
    """Update an existing dataset version."""
    logging.info(f"Starting dataset update process...")
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Current version: {current_version}")
    
    # Set up storage first
    if not setup_storage():
        logging.error("Failed to set up storage")
        return False
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    # Get initial file count
    initial_counts = count_files(dataset_dir)
    logging.info("\nInitial file counts:")
    logging.info(f"Total files: {initial_counts['total']}")
    logging.info(f"Images: {initial_counts['images']}")
    logging.info(f"Labels: {initial_counts['labels']}")
    for split, counts in initial_counts['by_split'].items():
        logging.info(f"{split}: {counts['images']} images, {counts['labels']} labels")
    
    # Create a mapping for class ID changes if new products are added
    class_id_mapping = {}
    if new_products:
        yaml_path = dataset_dir / "dataset.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
            
            # Get current classes
            current_classes = yaml_data.get("names", {})
            
            # Create a list of current class names
            current_class_names = []
            for class_id, class_name in current_classes.items():
                current_class_names.append(class_name)
            
            # Filter out new products that already exist
            unique_new_products = []
            for product in new_products:
                if product not in current_class_names:
                    unique_new_products.append(product)
                else:
                    logging.info(f"Product '{product}' already exists in dataset, skipping...")
            
            if not unique_new_products:
                logging.info("No new unique products to add.")
            else:
                # Create a list of all class names (existing + new unique)
                all_class_names = current_class_names.copy()
                all_class_names.extend(unique_new_products)
                
                # Sort alphabetically
                all_class_names.sort()
                
                # Create new class mapping
                new_class_mapping = {}
                for i, class_name in enumerate(all_class_names):
                    new_class_mapping[i] = class_name
                
                # Create mapping from old IDs to new IDs
                for old_id, old_name in current_classes.items():
                    for new_id, new_name in new_class_mapping.items():
                        if old_name == new_name:
                            class_id_mapping[int(old_id)] = new_id
                            break
                
                # Update yaml data with new class mapping
                yaml_data["names"] = new_class_mapping
                
                with open(yaml_path, "w") as f:
                    yaml.dump(yaml_data, f)
                logging.info(f"\nAdded {len(unique_new_products)} new unique products to dataset.yaml")
                logging.info(f"Updated class mapping: {class_id_mapping}")
    
    # If new data path is provided, copy new data
    if new_data_path:
        new_data_dir = Path(new_data_path)
        if not new_data_dir.exists():
            logging.error(f"New data directory not found: {new_data_path}")
            return False
            
        logging.info(f"\nCopying new data from: {new_data_path}")
        for split in ["train", "valid", "test"]:
            # Copy images
            src_img_dir = new_data_dir / split / "images"
            dst_img_dir = dataset_dir / split / "images"
            if src_img_dir.exists():
                for img_file in src_img_dir.glob("*.jpg"):
                    shutil.copy2(img_file, dst_img_dir)
                logging.info(f"Copied {len(list(dst_img_dir.glob('*.jpg')))} images to {split}/images")
            
            # Copy labels
            src_label_dir = new_data_dir / split / "labels"
            dst_label_dir = dataset_dir / split / "labels"
            if src_label_dir.exists():
                for label_file in src_label_dir.glob("*.txt"):
                    shutil.copy2(label_file, dst_label_dir)
                logging.info(f"Copied {len(list(dst_label_dir.glob('*.txt')))} labels to {split}/labels")
    
    # Update label files with new class IDs if class mapping exists
    if class_id_mapping:
        logging.info("\nUpdating label files with new class IDs...")
        for split in ["train", "valid", "test"]:
            label_dir = dataset_dir / split / "labels"
            if not label_dir.exists():
                continue
                
            updated_count = 0
            for label_file in label_dir.glob("*.txt"):
                updated = False
                with open(label_file, "r") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in class_id_mapping:
                            parts[0] = str(class_id_mapping[class_id])
                            new_lines.append(" ".join(parts) + "\n")
                            updated = True
                        else:
                            new_lines.append(line)
                
                if updated:
                    with open(label_file, "w") as f:
                        f.writelines(new_lines)
                    updated_count += 1
            
            logging.info(f"Updated {updated_count} label files in {split}/labels")
    
    # Add to DVC
    logging.info("\nAdding dataset to DVC...")
    if not run_command(f"dvc add {dataset_dir}"):
        logging.error("Failed to add dataset to DVC")
        return False
    
    # Add all files to Git
    logging.info("Adding files to Git...")
    if not run_command("git add ."):
        logging.error("Failed to add files to Git")
        return False
    
    # Commit changes
    commit_msg = f"Update dataset to version {output_version if output_version else 'latest'}"
    if not run_command(f'git commit -m "{commit_msg}"'):
        logging.error("Failed to commit changes")
        return False
    
    # Push to MinIO
    logging.info("Pushing to MinIO...")
    if not run_command("dvc push"):
        logging.error("Failed to push to MinIO")
        return False
    
    # Get final file count
    final_counts = count_files(dataset_dir)
    logging.info("\nFinal file counts:")
    logging.info(f"Total files: {final_counts['total']}")
    logging.info(f"Images: {final_counts['images']}")
    logging.info(f"Labels: {final_counts['labels']}")
    for split, counts in final_counts['by_split'].items():
        logging.info(f"{split}: {counts['images']} images, {counts['labels']} labels")
    
    logging.info(f"\nDataset successfully updated at: {dataset_dir}")
    logging.info("\nTo switch to a different version in the future, use:")
    logging.info("1. git checkout <commit-hash>")
    logging.info("2. dvc checkout")
    
    return True

def count_files(directory: Path) -> dict:
    """Count files in the dataset directory.
    
    Args:
        directory: Path to dataset directory
        
    Returns:
        Dictionary containing file counts
    """
    counts = {
        'total': 0,
        'images': 0,
        'labels': 0,
        'by_split': {}
    }
    
    for split in ["train", "valid", "test"]:
        split_dir = directory / split
        if not split_dir.exists():
            continue
            
        counts['by_split'][split] = {
            'images': 0,
            'labels': 0
        }
        
        # Count images
        img_dir = split_dir / "images"
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg")))
            counts['by_split'][split]['images'] = img_count
            counts['images'] += img_count
        
        # Count labels
        label_dir = split_dir / "labels"
        if label_dir.exists():
            label_count = len(list(label_dir.glob("*.txt")))
            counts['by_split'][split]['labels'] = label_count
            counts['labels'] += label_count
    
    counts['total'] = counts['images'] + counts['labels']
    return counts

def check_version(dataset_path: str) -> bool:
    """Check the current version of a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        bool: True if successful
    """
    logging.info(f"Checking dataset version at: {dataset_path}")
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_path)
    
    # Get file counts
    file_counts = count_files(dataset_dir)
    
    # Print dataset information
    logging.info("\nDataset Information:")
    logging.info("-" * 50)
    logging.info(f"Classes ({len(dataset_info['classes'])}):")
    for i, class_name in enumerate(dataset_info['classes']):
        logging.info(f"  {i}: {class_name}")
    
    logging.info("\nFile Counts:")
    logging.info(f"Total files: {file_counts['total']}")
    logging.info(f"Images: {file_counts['images']}")
    logging.info(f"Labels: {file_counts['labels']}")
    
    logging.info("\nSplit-wise Counts:")
    for split, counts in file_counts['by_split'].items():
        logging.info(f"{split}: {counts['images']} images, {counts['labels']} labels")
    
    # Check DVC status
    logging.info("\nDVC Status:")
    if not run_command("dvc status"):
        return False
    
    return True

def main():
    """Main function to handle dataset management."""
    parser = argparse.ArgumentParser(description="Manage dataset versions and updates")
    parser.add_argument(
        "--action",
        choices=["create", "update", "setup", "check"],
        required=True,
        help="Action to perform: create new dataset, update existing one, setup storage, or check version"
    )
    parser.add_argument(
        "--dataset-path",
        default="./data/datasets",
        help="Path to datasets directory"
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset (required for create action)"
    )
    parser.add_argument(
        "--current-version",
        help="Current version to update (required for update action)"
    )
    parser.add_argument(
        "--new-products",
        nargs="+",
        help="List of new product names to add"
    )
    parser.add_argument(
        "--new-data-path",
        help="Path to new data to add"
    )
    parser.add_argument(
        "--output-version",
        help="New version name for the dataset (used for both create and update actions)"
    )
    parser.add_argument(
        "--remote-url",
        help="Remote URL for DVC storage"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    if args.action == "setup":
        if not setup_storage():
            sys.exit(1)
    elif args.action == "create":
        if not args.dataset_name:
            logging.error("--dataset-name is required for create action")
            sys.exit(1)
        if not create_new_dataset(args.dataset_path, args.dataset_name, args.output_version):
            sys.exit(1)
    elif args.action == "update":
        if not args.current_version:
            logging.error("--current-version is required for update action")
            sys.exit(1)
        if not update_dataset(
            args.dataset_path,
            args.dataset_name,  # This can be None for update
            args.current_version,
            args.new_products,
            args.new_data_path,
            args.output_version,
            args.remote_url
        ):
            sys.exit(1)
    elif args.action == "check":
        if not check_version(args.dataset_path):
            sys.exit(1)

if __name__ == "__main__":
    main() 