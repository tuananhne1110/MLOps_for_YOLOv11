"""Utility functions for DVC dataset versioning."""
import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import shutil
import yaml

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DVC settings
DVC_REMOTE_NAME = os.getenv("DVC_REMOTE_NAME", "dvc")
DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "s3://dvc")
DVC_S3_ENDPOINT_URL = os.getenv("DVC_S3_ENDPOINT_URL", "http://localhost:9000")

logger = logging.getLogger(__name__)


class DVCClient:
    """Client for interacting with DVC."""
    
    def __init__(
        self,
        repo_dir: Union[str, Path],
        remote_name: str = DVC_REMOTE_NAME,
        remote_url: str = DVC_REMOTE_URL,
        s3_endpoint_url: str = DVC_S3_ENDPOINT_URL,
    ):
        """Initialize DVC client.
        
        Args:
            repo_dir: Path to repository directory
            remote_name: Name of DVC remote
            remote_url: URL of DVC remote
            s3_endpoint_url: S3 endpoint URL for custom S3 provider
        """
        self.repo_dir = Path(repo_dir)
        self.remote_name = remote_name
        self.remote_url = remote_url
        self.s3_endpoint_url = s3_endpoint_url
        
        # Ensure repository directory exists
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Git if not already initialized
        self._init_git()
        
        # Initialize DVC if not already initialized
        self._init_dvc()
    
    def _init_git(self) -> None:
        """Initialize Git repository if not already initialized."""
        try:
            # Check if Git is initialized
            if not os.path.exists(os.path.join(self.repo_dir, ".git")):
                self._run_command(["git", "init"])
                logging.info("Initialized Git repository")
                
                # Configure basic Git user info for commits
                self._run_command(["git", "config", "user.email", "dvc-user@example.com"])
                self._run_command(["git", "config", "user.name", "DVC User"])
                
                # Check if there are any files to commit
                result = self._run_command(["git", "status", "--porcelain"], check=False)
                if result.stdout.strip():
                    # There are changes to commit
                    self._run_command(["git", "add", "."])
                    self._run_command(["git", "commit", "-m", "Initial commit"])
                    logging.info("Created initial Git commit")
                else:
                    logging.info("No files to commit in Git repository")
                
            else:
                logging.info(f"Using existing Git repository in {self.repo_dir}")
                
        except Exception as e:
            logging.error(f"Error initializing Git: {e}")
            raise
    
    def _init_dvc(self) -> None:
        """Initialize DVC repository."""
        dvc_dir = self.repo_dir / ".dvc"
        if not dvc_dir.exists():
            logger.info(f"Initializing DVC repository in {self.repo_dir}")
            self._run_command(["dvc", "init"])
            
            # Configure DVC remote
            self._run_command([
                "dvc", "remote", "add", 
                "--default", self.remote_name, self.remote_url
            ])
            
            # Configure S3 endpoint
            self._run_command([
                "dvc", "remote", "modify", self.remote_name,
                "endpointurl", self.s3_endpoint_url
            ])
            
            logger.info(f"DVC repository initialized with remote {self.remote_name}")
        else:
            logger.info(f"Using existing DVC repository in {self.repo_dir}")
    
    def _run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command in the repository directory.
        
        Args:
            command: Command to run
            check: Whether to check the return code
            
        Returns:
            Completed process information
        """
        result = subprocess.run(
            command,
            cwd=self.repo_dir,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    
    def add_data(self, data_path: Union[str, Path]) -> bool:
        """Add data to DVC.
        
        Args:
            data_path: Path to data to add
            
        Returns:
            True if successful
        """
        try:
            data_path = Path(data_path)
            if not data_path.exists():
                logger.error(f"Data path does not exist: {data_path}")
                return False
            
            # Add data to DVC
            self._run_command(["dvc", "add", str(data_path)])
            logger.info(f"Added data to DVC: {data_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add data to DVC: {e}")
            return False
    
    def push_data(self) -> bool:
        """Push data to remote.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Pushing data to DVC remote")
            self._run_command(["dvc", "push"])
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push data to DVC remote: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    
    def pull_data(self, target_path: Optional[Union[str, Path]] = None) -> bool:
        """Pull data from remote.
        
        Args:
            target_path: Optional path to pull specific data
            
        Returns:
            True if successful
        """
        try:
            if target_path:
                logger.info(f"Pulling data from DVC remote: {target_path}")
                self._run_command(["dvc", "pull", str(target_path)])
            else:
                logger.info("Pulling all data from DVC remote")
                self._run_command(["dvc", "pull"])
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull data from DVC remote: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    
    def create_tag(self, tag_name: str, message: Optional[str] = None) -> bool:
        """Create a tag for the current state.
        
        Args:
            tag_name: Name of the tag
            message: Optional message for the tag
            
        Returns:
            True if successful
        """
        try:
            # First commit any changes to git
            msg = message or f"Create tag {tag_name}"
            self._run_command(["git", "add", "."])
            self._run_command(["git", "commit", "-m", msg])
            
            # Create the tag
            logger.info(f"Creating tag: {tag_name}")
            self._run_command(["git", "tag", "-a", tag_name, "-m", msg])
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create tag: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    
    def checkout_tag(self, tag_name: str) -> bool:
        """Checkout a specific tag.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Checking out tag: {tag_name}")
            self._run_command(["git", "checkout", tag_name])
            self._run_command(["dvc", "checkout"])
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout tag: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False
    
    def create_branch(self, branch_name: str) -> bool:
        """Create a new branch.
        
        Args:
            branch_name: Name of the branch to create
            
        Returns:
            True if successful
        """
        try:
            # Create and switch to new branch
            self._run_command(["git", "checkout", "-b", branch_name])
            logger.info(f"Created branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def checkout_branch(self, branch_name: str) -> bool:
        """Checkout an existing branch.
        
        Args:
            branch_name: Name of the branch to checkout
            
        Returns:
            True if successful
        """
        try:
            self._run_command(["git", "checkout", branch_name])
            logger.info(f"Checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch: {e}")
            return False


def get_dataset_path(dataset_name: str, repo_dir: Union[str, Path] = None) -> Path:
    """Get path to dataset in the repository.
    
    Args:
        dataset_name: Name of the dataset
        repo_dir: Optional repository directory
        
    Returns:
        Path to dataset
    """
    if repo_dir is None:
        repo_dir = Path("data/dvc_repo")
    else:
        repo_dir = Path(repo_dir)
    
    return repo_dir / "datasets" / dataset_name 