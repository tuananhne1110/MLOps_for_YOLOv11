#!/usr/bin/env python3
"""Initialize DVC repository and set up directories."""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.dvc_utils import DVCClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Initialize DVC repository")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="data/dvc_repo",
        help="Path to DVC repository directory"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="data/datasets",
        help="Path to datasets directory to be tracked"
    )
    parser.add_argument(
        "--remote-name",
        type=str,
        default=os.getenv("DVC_REMOTE_NAME", "dvc"),
        help="Name of DVC remote"
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=os.getenv("DVC_REMOTE_URL", "s3://dvc"),
        help="URL of DVC remote"
    )
    parser.add_argument(
        "--s3-endpoint-url",
        type=str,
        default=os.getenv("DVC_S3_ENDPOINT_URL", "http://localhost:9000"),
        help="S3 endpoint URL for custom S3 provider"
    )
    parser.add_argument(
        "--link-datasets",
        action="store_true",
        help="Create symbolic links to datasets directory"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create paths
    repo_dir = Path(args.repo_dir)
    datasets_dir = Path(args.datasets_dir)
    
    logger.info(f"Initializing DVC repository in {repo_dir}")
    logger.info(f"Using datasets from {datasets_dir}")
    
    # Create DVC client
    dvc_client = DVCClient(
        repo_dir=repo_dir,
        remote_name=args.remote_name,
        remote_url=args.remote_url,
        s3_endpoint_url=args.s3_endpoint_url
    )
    
    # Create datasets directory in repo if not exists
    repo_datasets_dir = repo_dir / "datasets"
    repo_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    if args.link_datasets and datasets_dir.exists():
        # Create symbolic links to datasets
        for dataset_path in datasets_dir.iterdir():
            if dataset_path.is_dir():
                dataset_name = dataset_path.name
                target_path = repo_datasets_dir / dataset_name
                
                if not target_path.exists():
                    logger.info(f"Creating symbolic link for dataset: {dataset_name}")
                    
                    # Create relative symlink
                    rel_path = os.path.relpath(dataset_path, target_path.parent)
                    os.symlink(rel_path, target_path)
                    
                    # Add to DVC
                    dvc_client.add_data(f"datasets/{dataset_name}")
                else:
                    logger.info(f"Dataset already exists in repo: {dataset_name}")
        
        # Push to remote
        logger.info("Pushing data to remote")
        dvc_client.push_data()
    else:
        logger.info("No datasets were linked. Repository is ready for use.")
    
    logger.info(f"DVC repository initialized at {repo_dir}")
    logger.info(f"To add new datasets, place them in {datasets_dir} and run this script again with --link-datasets")


if __name__ == "__main__":
    main() 