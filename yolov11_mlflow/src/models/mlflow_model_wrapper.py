"""MLflow model wrapper for YOLOv11."""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import mlflow.pyfunc
from ultralytics import YOLO
import torch


class YOLOModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for YOLOv11 model to be used with MLflow."""
    
    def __init__(self, model_path: str):
        """Initialize the model wrapper.
        
        Args:
            model_path: Path to the YOLO model file
        """
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the model from the context.
        
        Args:
            context: MLflow model context containing artifacts
        """
        # Get the model path from data_path
        model_path = context.data_path
        self.model = YOLO(model_path)
        self.model.to(self.device)
    
    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the model.
        
        Args:
            context: MLflow model context
            model_input: Dictionary containing input data and parameters
            
        Returns:
            Dictionary containing model predictions
        """
        if self.model is None:
            self.load_context(context)
        
        # Extract parameters from input
        image = model_input.get("image")
        conf_threshold = model_input.get("conf_threshold", 0.25)
        iou_threshold = model_input.get("iou_threshold", 0.45)
        
        if image is None:
            raise ValueError("No image provided in model_input")
        
        # Run inference
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Process results
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                prediction = {
                    "bbox": box.xyxy[0].tolist(),  # Convert to list for JSON serialization
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])]
                }
                predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "image_shape": image.shape if isinstance(image, np.ndarray) else None
        }


def _load_pyfunc(model_path: str) -> YOLOModelWrapper:
    """Load the model wrapper.
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        YOLOModelWrapper instance
    """
    return YOLOModelWrapper(model_path) 