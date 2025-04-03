# MLOps for YOLOv11

This project implements a complete MLOps pipeline for training and deploying YOLOv11 object detection models. It provides a robust infrastructure for model training, experiment tracking, and model versioning using industry-standard tools and practices.

## Features

- **MLOps Infrastructure**:
  - MLflow for experiment tracking and model registry
  - DVC for data versioning and pipeline management
  - MinIO for object storage (S3-compatible)
  - PostgreSQL for MLflow backend storage
  - Docker-based deployment

- **Training Pipeline**:
  - YOLOv11 model training with configurable parameters
  - Comprehensive data augmentation options
  - Automatic model checkpointing and versioning
  - GPU support with configurable device selection
  - Experiment tracking with MLflow

- **Configuration Management**:
  - YAML-based configuration for training parameters
  - Environment variable management
  - Flexible model architecture selection (yolov11n, yolov11s, yolov11m, yolov11l, yolov11x)

## How It Works

This project provides a complete workflow for training and managing YOLOv11 object detection models. Here's how it works:

1. **Data Management**:
   - Your training data is stored in MinIO (like AWS S3)
   - DVC helps track different versions of your dataset
   - This ensures you can always reproduce your training results

2. **Training Process**:
   - Configure your training parameters in `configs/training_config.yaml`
   - Run the training script
   - The system automatically:
     - Tracks all training metrics in MLflow
     - Saves model checkpoints
     - Logs training progress
     - Manages GPU resources

3. **Experiment Tracking**:
   - MLflow keeps track of:
     - Training metrics (accuracy, loss, etc.)
     - Model parameters
     - Training configurations
     - Model artifacts
   - You can compare different training runs
   - Access results through MLflow's web interface

4. **Infrastructure**:
   - Everything runs in Docker containers
   - PostgreSQL stores MLflow data
   - MinIO stores large files (datasets, models)
   - All services are automatically managed

5. **Model Management**:
   - Different model sizes available (nano to extra-large)
   - Automatic checkpointing during training
   - Easy model versioning and comparison
   - Simple deployment process

## Detailed Pipeline

### 1. Data Pipeline
```
Raw Data → DVC Versioning → MinIO Storage → Training Pipeline
```
- **Data Collection**: Gather your object detection dataset
- **Data Versioning**: 
  - Use DVC to track dataset changes
  - Store data in MinIO (S3-compatible storage)
  - Maintain data lineage and reproducibility
- **Data Preparation**:
  - Automatic data validation
  - Format conversion if needed
  - Dataset splitting (train/val/test)

### 2. Training Pipeline
```
Configuration → Model Training → Checkpointing → Experiment Tracking
```
- **Configuration**:
  - Set model parameters in `training_config.yaml`
  - Choose model architecture (yolov11n to yolov11x)
  - Configure training hyperparameters
  - Set up data augmentation

- **Training Process**:
  - Automatic GPU detection and utilization
  - Progress monitoring and logging
  - Real-time metric tracking
  - Automatic checkpointing

- **Experiment Tracking**:
  - MLflow integration for metric logging
  - Parameter tracking
  - Artifact storage
  - Model versioning

### 3. MLOps Infrastructure
```
Docker Services → MLflow → PostgreSQL → MinIO
```
- **Docker Services**:
  - MLflow server for experiment tracking
  - PostgreSQL for metadata storage
  - MinIO for artifact storage
  - All services containerized for easy deployment

- **Monitoring and Management**:
  - MLflow UI for experiment visualization
  - MinIO console for artifact management
  - PostgreSQL for data persistence
  - Automatic service health checks

### 4. Model Deployment Pipeline
```
Model Registry → Version Control → Deployment
```
- **Model Registry**:
  - Store trained models in MLflow
  - Version control for model artifacts
  - Model metadata tracking
  - Performance metrics storage

- **Deployment Process**:
  - Model version selection
  - Environment setup
  - Inference service deployment
  - Performance monitoring

### 5. Development Workflow
```
Code Changes → Pre-commit Hooks → Testing → Deployment
```
- **Development Process**:
  - Code version control with Git
  - Pre-commit hooks for code quality
  - Automated testing
  - CI/CD pipeline integration

- **Quality Assurance**:
  - Code formatting checks
  - Unit testing
  - Integration testing
  - Performance benchmarking

## Project Structure

```
.
├── configs/                 # Configuration files
│   └── training_config.yaml # Training parameters
├── src/                    # Source code
│   ├── training/          # Training scripts
│   ├── inference/         # Inference code
│   ├── models/           # Model definitions
│   └── utils/            # Utility functions
├── scripts/               # Utility scripts
├── docker-compose.yml     # Docker services configuration
├── requirements.txt       # Python dependencies
└── .env                  # Environment variables
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- CUDA-capable GPU (recommended)
- AWS credentials (for S3 storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tuananhne1110/MLOps_for_YOLOv11.git
cd MLOps_for_YOLOv11
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start the MLOps infrastructure:
```bash
python script/start_services.py 
```

## Usage

### Training

To train a YOLOv11 model:

```bash
python src/training/train.py --config configs/training_config.yaml
```

The training script will:
- Load the configuration
- Set up MLflow tracking
- Train the model with specified parameters
- Save checkpoints and model artifacts
- Log metrics and parameters to MLflow

### Configuration

Edit `configs/training_config.yaml` to customize:
- Model architecture and size
- Training parameters (batch size, epochs, etc.)
- Data augmentation settings
- Optimizer configuration
- MLflow experiment settings

## MLOps Infrastructure

The project uses the following services:

- **MLflow**: Experiment tracking and model registry
  - Access at: http://localhost:5000
  - Tracks metrics, parameters, and artifacts

- **MinIO**: S3-compatible object storage
  - Access at: http://localhost:9000
  - Stores model artifacts and data

- **PostgreSQL**: MLflow backend storage
  - Stores experiment metadata and metrics

