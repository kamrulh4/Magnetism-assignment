import os
import logging
import torch

# Patch torch.load to default to weights_only=False to support PyTorch 2.6+ loading for YOLO models
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YOLOv8-API")

# Lazy-loaded YOLOv8 model cached in memory
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            logger.info("Loading YOLOv8n model...")
            # Automatically downloads weights (yolov8n.pt) if not present
            _model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load YOLO model: {str(e)}")
    return _model
