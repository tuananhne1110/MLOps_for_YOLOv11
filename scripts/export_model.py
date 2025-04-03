#!/usr/bin/env python
"""Script to export trained YOLOv11 model to different formats."""
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Union, List
from copy import deepcopy

import mlflow
import torch
import torch.nn as nn
import onnx
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.utils.torch_utils import select_device

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

def suppress_warnings():
    """Suppress unnecessary warnings."""
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

class DeepStreamOutput(nn.Module):
    """DeepStream output layer for YOLOv11."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        det_boxes = x[:, :, :4]
        det_scores, det_classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        det_classes = det_classes.float()
        return det_boxes, det_scores, det_classes

def prepare_yolov11_model(model_path: str, device: str = "cpu"):
    """Prepare YOLOv11 model for export.
    
    Args:
        model_path: Path to model weights
        device: Device to load model on
    """
    model = YOLO(model_path)
    model = deepcopy(model.model).to(device)
    
    # Prepare model for export
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    
    # Configure detection layers
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    
    return model

def export_deepstream(
    model_path: str,
    img_size: List[int] = [640],
    batch_size: int = 1,
    dynamic: bool = False,
    opset: int = 16,
    simplify: bool = True,
):
    """Export model to DeepStream-compatible ONNX format.
    
    Args:
        model_path: Path to model weights
        img_size: Input image size [H,W]
        batch_size: Batch size for static export
        dynamic: Whether to use dynamic batch size
        opset: ONNX opset version
        simplify: Whether to simplify ONNX model
    """
    suppress_warnings()
    device = select_device("cpu")
    
    # Prepare model
    model = prepare_yolov11_model(model_path, device)
    
    # Create labels file
    if len(model.names.keys()) > 0:
        logging.info("Creating labels.txt file")
        with open("labels.txt", "w") as f:
            for name in model.names.values():
                f.write(name + "\n")
    
    # Add DeepStream output layer
    model = nn.Sequential(model, DeepStreamOutput())
    
    # Prepare input tensor
    img_size = img_size * 2 if len(img_size) == 1 else img_size
    onnx_input_im = torch.zeros(batch_size, 3, *img_size).to(device)
    
    # Set up dynamic axes
    dynamic_axes = {
        "input": {0: "batch"},
        "boxes": {0: "batch"},
        "scores": {0: "batch"},
        "classes": {0: "batch"},
    } if dynamic else None
    
    # Export to ONNX
    onnx_output_file = os.path.basename(model_path).split(".pt")[0] + "_deepstream.onnx"
    logging.info(f"Exporting model to ONNX: {onnx_output_file}")
    
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "scores", "classes"],
        dynamic_axes=dynamic_axes,
    )
    
    # Simplify ONNX model if requested
    if simplify:
        logging.info("Simplifying ONNX model")
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)
    
    return onnx_output_file

def export_model(run_id: str, format: str = "onnx", half: bool = True, deepstream: bool = False):
    """Export model to specified format.
    
    Args:
        run_id: MLflow run ID containing the model
        format: Export format (onnx, engine, tflite, etc.)
        half: Whether to export in FP16 (half precision)
        deepstream: Whether to export in DeepStream format
    """
    # Get model from MLflow
    model_uri = f"runs:/{run_id}/best_model"
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="best_model")
    
    if deepstream:
        # Export in DeepStream format
        output_file = export_deepstream(
            local_path,
            img_size=[640],
            batch_size=1,
            dynamic=True,
            opset=16,
            simplify=True,
        )
        logging.info(f"Model exported successfully to {output_file}")
    else:
        # Load model and export in standard format
        model = YOLO(local_path)
        logging.info(f"Exporting model to {format} format...")
        model.export(
            format=format,
            half=half,
            dynamic=True,
            simplify=True,
        )
        logging.info(f"Model exported successfully to {model.export_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to different formats")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID containing the model"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "engine", "tflite", "saved_model", "pb", "torchscript"],
        help="Export format"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 (half precision)"
    )
    parser.add_argument(
        "--deepstream",
        action="store_true",
        help="Export in DeepStream format"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        export_model(args.run_id, args.format, args.half, args.deepstream)
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 