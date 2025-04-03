"""DVC filesystem implementation for accessing data during training."""
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import subprocess
import shutil

from .dvc_utils import DVCClient

logger = logging.getLogger(__name__)


class DVCFileHandler:
    """Handler for working with DVC filesystem.
    
    This class enables accessing data in DVC repositories during training.
    """
    
    def __init__(self):
        """Initialize the DVC file handler."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dvc_"))
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Default repo directory
        repo_dir = os.getenv("DVC_REPO_DIR", Path("data/dvc_repo"))
        self.dvc_client = DVCClient(repo_dir=repo_dir)
        
        # Keep track of datasets pulled from DVC
        self.pulled_datasets = set()
        
        logger.info(f"Initialized DVC file handler with cache dir: {self.cache_dir}")
    
    def get_local_path(self, dataset_path: Union[str, Path], pull: bool = True) -> Path:
        """Get a local path for a dataset.
        
        Args:
            dataset_path: Path to dataset directory
            pull: Whether to pull data from remote
            
        Returns:
            Local path to dataset
        """
        dataset_path = Path(dataset_path)
        
        # Check if already in local filesystem
        if dataset_path.exists():
            if dataset_path in self.pulled_datasets:
                logger.info(f"Using cached dataset: {dataset_path}")
                return dataset_path
            
            # Add to pulled datasets
            self.pulled_datasets.add(dataset_path)
            
            if pull:
                # Pull latest data from DVC remote
                logger.info(f"Pulling latest data for: {dataset_path}")
                self.dvc_client.pull_data()
            
            return dataset_path
        else:
            # Try as a relative path inside the DVC repository
            repo_dataset_path = Path(self.dvc_client.repo_dir) / dataset_path
            
            if repo_dataset_path.exists():
                if repo_dataset_path in self.pulled_datasets:
                    logger.info(f"Using cached dataset: {repo_dataset_path}")
                    return repo_dataset_path
                
                # Add to pulled datasets
                self.pulled_datasets.add(repo_dataset_path)
                
                if pull:
                    # Pull latest data from DVC remote
                    logger.info(f"Pulling latest data for: {repo_dataset_path}")
                    rel_path = repo_dataset_path.relative_to(self.dvc_client.repo_dir)
                    self.dvc_client.pull_data(rel_path)
                
                return repo_dataset_path
            
            # Dataset not found
            logger.error(f"Dataset not found in DVC repository: {dataset_path}")
            return dataset_path
    
    def checkout_dataset_branch(self, dataset_name: str, branch: str = None) -> Path:
        """Checkout a specific branch for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            branch: Branch name (default: 'dataset/{dataset_name}')
            
        Returns:
            Path to dataset
        """
        if branch is None:
            branch = f"dataset/{dataset_name}"
        
        logger.info(f"Checking out branch {branch} for dataset {dataset_name}")
        
        # Checkout the branch
        self.dvc_client.checkout_branch(branch)
        
        # Pull the data
        self.dvc_client.pull_data()
        
        # Get the dataset path
        dataset_path = Path(self.dvc_client.repo_dir) / "datasets" / dataset_name
        
        # Add to pulled datasets
        self.pulled_datasets.add(dataset_path)
        
        return dataset_path
    
    def checkout_dataset_tag(self, dataset_name: str, tag: str) -> Path:
        """Checkout a specific tag for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            tag: Tag name
            
        Returns:
            Path to dataset
        """
        logger.info(f"Checking out tag {tag} for dataset {dataset_name}")
        
        # Checkout the tag
        self.dvc_client.checkout_tag(tag)
        
        # Pull the data
        self.dvc_client.pull_data()
        
        # Get the dataset path
        dataset_path = Path(self.dvc_client.repo_dir) / "datasets" / dataset_name
        
        # Add to pulled datasets
        self.pulled_datasets.add(dataset_path)
        
        return dataset_path
    
    def cleanup(self):
        """Clean up temporary files."""
        logger.info(f"Cleaning up DVC temporary directory: {self.temp_dir}")
        
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Failed to clean up temporary directory: {e}")


# Singleton instance
dvc_handler = DVCFileHandler()


def resolve_dvc_path(path: Union[str, Path]) -> Path:
    """Resolve a DVC path to a local path.
    
    Args:
        path: Path to resolve, can be a local path or dataset name
        
    Returns:
        Resolved local path
    """
    path = Path(path)
    
    # If it's already a full path that exists, return it
    if path.exists() and path.is_absolute():
        return path
    
    # Try to resolve as a dataset path in the DVC repo
    return dvc_handler.get_local_path(path)


def cleanup_dvc():
    """Clean up DVC temporary files."""
    dvc_handler.cleanup() 