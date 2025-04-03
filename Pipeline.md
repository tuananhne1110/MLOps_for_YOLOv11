# YOLOv11 ML Pipeline Documentation

## Project Structure
```
yolov11_mlflow/
├── configs/           # Configuration files
├── data/             # Dataset and DVC repository
├── models/           # Model weights and checkpoints
├── runs/             # Training runs and outputs
├── scripts/          # Utility scripts
├── src/              # Source code
│   ├── training/     # Training related code
│   └── utils/        # Utility functions
├── .env              # Environment variables
├── docker-compose.yml # Docker services configuration
└── requirements.txt  # Python dependencies
```

## Pipeline Workflow

### Main Functions and Flow

#### 1. Service Management (`scripts/start_services.py`)
```python
def start_services():
    """Start MLflow and related services using Docker Compose."""
    # 1. Check Docker installation
    # 2. Start services (PostgreSQL, MinIO, MLflow)
    # 3. Wait for services to be ready
    # 4. Initialize DVC repository
```

#### 2. Training Pipeline (`src/training/train.py`)
```python
def train_yolov11(config_path: str):
    """Main training function with MLflow tracking."""
    # 1. Load configurations
    config = load_yaml_config(config_path)
    training_config = config["training"]
    dataset_config = config["dataset"]
    mlflow_config = config["mlflow"]
    
    # 2. Set up MLflow tracking
    mlflow_settings = get_mlflow_config()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_settings["s3_endpoint_url"]
    os.environ["MLFLOW_S3_BUCKET_NAME"] = mlflow_settings["s3_bucket_name"]
    
    # 3. Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name=mlflow_config["experiment_name"],
        run_name=mlflow_config["run_name"],
        tags=mlflow_config["tags"],
    )
    
    # 4. Start training process
    with mlflow.start_run():
        # 4.1 Load model
        model = YOLO(str(model_path))
        
        # 4.2 Prepare dataset
        data_yaml = prepare_dataset_config(dataset_config["data_yaml"])
        
        # 4.3 Train model
        results = model.train(**args)
        
        # 4.4 Log metrics and artifacts
        mlflow_logger.log_metrics(metrics)
        mlflow_logger.log_artifacts(artifacts)
```

#### 3. Data Management (`src/utils/dvc_fs.py`)
```python
def resolve_dvc_path(path: str) -> Path:
    """Resolve DVC path to local filesystem."""
    # 1. Check if path is DVC-tracked
    # 2. If yes, resolve to local path
    # 3. If no, return original path
```

#### 4. MLflow Integration (`src/utils/mlflow_logger.py`)
```python
class MLflowLogger:
    """MLflow logging wrapper."""
    def __init__(self, experiment_name: str, run_name: str, tags: Dict):
        # Initialize MLflow tracking
        
    def start_run(self):
        # Start new MLflow run
        
    def log_params(self, params: Dict):
        # Log training parameters
        
    def log_metrics(self, metrics: Dict):
        # Log training metrics
        
    def log_artifact(self, artifact_path: str):
        # Log model artifacts
```

### Pipeline Execution Flow

1. **Initialization Phase**
   ```mermaid
   graph TD
   A[Start Services] --> B[Check Docker]
   B --> C[Start PostgreSQL]
   C --> D[Start MinIO]
   D --> E[Start MLflow]
   E --> F[Initialize DVC]
   ```

2. **Training Phase**
   ```mermaid
   graph TD
   A[Load Configs] --> B[Setup MLflow]
   B --> C[Load Model]
   C --> D[Prepare Dataset]
   D --> E[Training Loop]
   E --> F[Log Metrics]
   F --> G[Save Artifacts]
   ```

3. **Data Flow**
   ```mermaid
   graph LR
   A[Dataset] --> B[DVC]
   B --> C[Training]
   C --> D[MLflow]
   D --> E[MinIO]
   ```

### Key Components Interaction

1. **Data Management**
   - DVC handles dataset versioning
   - MinIO provides S3-compatible storage
   - Dataset configuration in YAML format

2. **Training Process**
   - YOLOv11 model initialization
   - Training loop with MLflow tracking
   - Checkpoint saving and validation

3. **Experiment Tracking**
   - MLflow for metrics and parameters
   - MinIO for artifact storage
   - PostgreSQL for metadata storage

4. **Model Management**
   - Checkpoint saving
   - Model artifact logging
   - Performance metrics tracking

### Error Handling and Recovery

1. **Service Failures**
   ```python
   try:
       start_services()
   except Exception as e:
       logger.error(f"Failed to start services: {e}")
       cleanup_services()
   ```

2. **Training Errors**
   ```python
   try:
       results = model.train(**args)
   except Exception as e:
       logger.error(f"Training failed: {e}")
       save_checkpoint(model)
   ```

3. **MLflow Integration**
   ```python
   try:
       mlflow_logger.log_artifact(artifact_path)
   except Exception as e:
       logger.error(f"Failed to log artifact: {e}")
       retry_logging()
   ```

## Pipeline Components

### 1. Environment Setup

#### 1.1 Docker Services
The project uses Docker Compose to manage the following services:
- PostgreSQL: MLflow backend database
- MinIO: S3-compatible storage for artifacts
- MLflow: Experiment tracking server
- DVC: Data version control

To start the services:
```bash
python scripts/start_services.py
```

To stop the services:
```bash
python scripts/start_services.py --stop
```

#### 1.2 Environment Variables
Key environment variables in `.env`:
```
# MLflow settings
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=yolov11-training

# MinIO settings
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_BUCKET_NAME=dvc
```

### 2. Data Management

#### 2.1 Dataset Structure
```
data/
├── datasets/
│   ├── train/       # Training images and labels
│   └── valid/       # Validation images and labels
└── dvc_repo/        # DVC repository for data versioning
```

#### 2.2 Dataset Configuration
Located in `data/dataset.yaml`:
```yaml
path: data/datasets  # Dataset root directory
train: train/images  # Training images
val: valid/images    # Validation images
names:               # Class names
  - class1
  - class2
  ...
```

### 3. Model Training

#### 3.1 Training Configuration
Located in `configs/training_config.yaml`:
```yaml
training:
  model_name: yolov11n
  epochs: 100
  batch_size: 4
  img_size: 640
  device: cuda
  workers: 4
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.0005
  augmentation:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 0.0
    translate: 0.1
    scale: 0.5
    shear: 0.0
    fliplr: 0.5
    mosaic: 0.0
    mixup: 0.0
```

#### 3.2 Training Process
1. Load configuration from YAML files
2. Set up MLflow tracking
3. Initialize YOLOv11 model
4. Train model with specified parameters
5. Log metrics and artifacts to MLflow
6. Save model checkpoints

To start training:
```bash
python src/training/train.py
```

### 4. Experiment Tracking

#### 4.1 MLflow Integration
- Tracks training metrics (precision, recall, mAP)
- Logs model artifacts
- Stores training parameters
- Saves training plots

#### 4.2 Artifact Storage
- Model weights stored in MinIO
- Training plots and metrics stored in MLflow
- Dataset versioned with DVC

### 5. Model Management

#### 5.1 Model Checkpoints
- Best model saved as `best.pt`
- Latest model saved as `last.pt`
- Checkpoints stored in `runs/<run_name>/weights/`

#### 5.2 Model Artifacts
- Model weights
- Training plots
- Configuration files
- Dataset statistics

### 6. Monitoring and Logging

#### 6.1 Training Logs
- Logs stored in `logs/training.log`
- Includes training progress and errors

#### 6.2 MLflow UI
- Access at http://localhost:5000
- View experiments and runs
- Compare model performance
- Download artifacts

### 7. Deployment Considerations

#### 7.1 Model Export
- Models can be exported to various formats (ONNX, TorchScript)
- Supports different inference backends

#### 7.2 Performance Metrics
- mAP50 and mAP50-95 for detection accuracy
- Inference speed metrics
- Memory usage tracking

## Usage Examples

### Starting the Pipeline
```bash
# 1. Start services
python scripts/start_services.py

# 2. Start training
python src/training/train.py

# 3. Monitor training
# Access MLflow UI at http://localhost:5000
```

### Stopping the Pipeline
```bash
# Stop all services
python scripts/start_services.py --stop
```

## Troubleshooting

### Common Issues
1. S3 Bucket Errors
   - Ensure MinIO is running
   - Check bucket creation in `createbuckets` service
   - Verify environment variables

2. MLflow Connection Issues
   - Check PostgreSQL connection
   - Verify MLflow server is running
   - Check network connectivity

3. Training Errors
   - Verify dataset structure
   - Check GPU memory usage
   - Validate configuration files

## Best Practices

1. Data Management
   - Use DVC for dataset versioning
   - Keep dataset structure consistent
   - Regular data validation

2. Training
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Regular checkpointing

3. Experiment Tracking
   - Meaningful run names
   - Comprehensive parameter logging
   - Regular artifact cleanup

4. Model Management
   - Version control for models
   - Regular model evaluation
   - Performance benchmarking 