# DishScan - YOLOv11-based Product Recognition System

A production-ready system for detecting and classifying different types of bread and pastries using YOLOv11, MLflow for experiment tracking, and DVC for dataset versioning.

## Overview

DishScan is a computer vision system designed to:
1. Detect and classify different types of bread and pastries in images
2. Track experiments and model performance using MLflow
3. Manage dataset versions using DVC and MinIO
4. Provide a scalable infrastructure for model training

## Features

- **Object Detection**: Using YOLOv11 for accurate product detection
- **Experiment Tracking**: MLflow integration for monitoring training metrics
- **Dataset Versioning**: DVC with MinIO storage for dataset management
- **Model Management**: Version control for model checkpoints
- **Docker Support**: Containerized services for easy deployment

## Project Structure

```
yolov11_mlflow/
├── configs/                 # Configuration files
│   └── training_config.yaml # Training parameters
├── data/                    # Data directory
│   ├── datasets/           # Raw datasets
│   └── dvc_repo/          # Version-controlled datasets
├── models/                 # Model checkpoints
├── scripts/               # Utility scripts
│   ├── dataset_manager.py # Dataset management with DVC
│   └── start_services.py  # Start MLflow and MinIO services
├── src/                   # Source code
│   └── training/         # Training code
├── .dvc/                 # DVC configuration
├── .env                  # Environment variables
└── docker-compose.yml    # Docker services configuration
```

## System Architecture

### 1. Data Management Layer
- **DVC**: Version control for datasets
- **MinIO**: S3-compatible storage for large files
- **Dataset Manager**: Script for managing dataset versions

### 2. Training Layer
- **YOLOv11**: Object detection model
- **MLflow**: Experiment tracking and model registry
- **PyTorch**: Deep learning framework

## Setup Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Tools
```bash
# Install MinIO Client (mc)
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Install DVC with S3 support
pip install dvc[s3]

# Install Git
sudo apt-get update
sudo apt-get install git
```

### 3. Start Services
```bash
# Start MinIO and MLflow
docker-compose up -d
```

## Usage Guide

### 1. Dataset Management

#### Initial Setup
```bash
# Initialize DVC and configure storage
python scripts/dataset_manager.py --action setup
```

#### Create New Dataset
```bash
python scripts/dataset_manager.py --action create \
    --dataset-path ./data/datasets \
    --dataset-name bread_dataset
```

#### Update Dataset
```bash
# Add new data
python scripts/dataset_manager.py --action update \
    --dataset-path ./data/datasets \
    --current-version v1.0 \
    --new-data-path /path/to/new/data

# Add new products
python scripts/dataset_manager.py --action update \
    --dataset-path ./data/datasets \
    --current-version v1.0 \
    --new-products "new_product1" "new_product2"
```

### 2. Model Training

```bash
# Train model
python src/training/train.py --config configs/training_config.yaml
```

Training configuration includes:
- Model architecture
- Training parameters
- Data augmentation
- Validation settings

## Monitoring and Management

### MLflow UI
- Access at http://localhost:5000
- View training metrics
- Compare experiments
- Download model artifacts

### MinIO Console
- Access at http://localhost:9001
- Manage dataset versions
- View model artifacts

## Dataset Information

### Current Classes
1. butter_sugar_bread
2. chicken_floss_bread
3. chicken_floss_sandwich
4. cream_puf
5. croissant
6. donut
7. muffin
8. salted_egg_sponge_cake
9. sandwich
10. sponge_cake
11. tiramisu

### Dataset Structure
```
data/datasets/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Best Practices

1. **Dataset Management**:
   - Use DVC for version control
   - Maintain consistent data structure
   - Document dataset changes

2. **Model Training**:
   - Monitor training metrics
   - Use validation data
   - Implement early stopping

3. **Version Control**:
   - Track dataset versions
   - Document model changes
   - Maintain experiment history

## Troubleshooting

1. **DVC Issues**:
   - Check DVC initialization
   - Verify MinIO connection
   - Ensure proper Git setup
   - If DVC push fails:
     ```bash
     # Check DVC status
     dvc status
     
     # Force push if needed
     dvc push -f
     ```

2. **Git Issues**:
   - If Git commit fails:
     ```bash
     # Check Git status
     git status
     
     # Configure Git if not done
     git config --global user.name "Your Name"
     git config --global user.email "your.email@example.com"
     
     # Force add DVC files if needed
     git add -f data/datasets.dvc
     
     # Try commit again
     git commit -m "Update dataset version"
     ```
   - If Git push fails:
     ```bash
     # Check remote configuration
     git remote -v
     
     # Force push if needed
     git push -f origin main
     ```

3. **Training Issues**:
   - Check GPU availability
   - Verify dataset format
   - Monitor memory usage
   - If training fails:
     ```bash
     # Check CUDA availability
     python -c "import torch; print(torch.cuda.is_available())"
     
     # Check GPU memory
     nvidia-smi
     ```

4. **MinIO Issues**:
   - If MinIO connection fails:
     ```bash
     # Check MinIO status
     docker-compose ps
     
     # Restart MinIO if needed
     docker-compose restart minio
     
     # Check MinIO logs
     docker-compose logs minio
     ```
   - Verify MinIO credentials in `.env`:
     ```
     MINIO_ROOT_USER=minioadmin
     MINIO_ROOT_PASSWORD=minioadmin
     ```

5. **Dataset Update Issues**:
   - If update fails:
     ```bash
     # Check dataset structure
     ls -R data/datasets/
     
     # Verify DVC tracking
     dvc status
     
     # Check Git tracking
     git status
     ```
   - Common problems:
     - Incorrect dataset path
     - Missing version information
     - Permission issues
     - Disk space issues

6. **Environment Issues**:
   - If dependencies are missing:
     ```bash
     # Update pip
     pip install --upgrade pip
     
     # Install requirements
     pip install -r requirements.txt
     
     # Check Python version
     python --version
     ```
   - If virtual environment issues:
     ```bash
     # Create new venv
     python -m venv venv
     
     # Activate venv
     source venv/bin/activate  # Linux/Mac
     .\venv\Scripts\activate   # Windows
     ```

7. **General Tips**:
   - Always check logs for detailed error messages
   - Ensure all services are running before operations
   - Keep sufficient disk space for operations
   - Regular backups of important data
   - Monitor system resources during operations

## Dependencies

- Python 3.8+
- PyTorch
- Ultralytics YOLOv11
- MLflow
- DVC
- MinIO
- Docker

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]
