import os
import io
import cv2
import numpy as np
import base64
import tempfile
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

# Patch torch.load to default to weights_only=False to support PyTorch 2.6+ loading for YOLO models
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YOLOv8-API")

# Lazy loading of YOLO model
model = None

def get_model():
    global model
    if model is None:
        try:
            logger.info("Loading YOLOv8n model...")
            # Automatically downloads weights (yolov8n.pt) if not present
            model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load YOLO model: {str(e)}")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield
    global model
    model = None

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="A FastAPI backend serving YOLOv8n object detection on images and videos",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path for saving static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "YOLOv8n"}

@app.post("/detect-image/")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read file contents
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")

        # Run inference
        yolo_model = get_model()
        results = yolo_model(img, conf=conf)
        
        # Parse detections
        detections = []
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            detections.append({
                "bbox": [round(x, 2) for x in xyxy],
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name
            })
            
        # Draw bounding boxes on the image using Ultralytics plot() helper
        annotated_img = result.plot()
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/detect-video/")
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    # Write temporary file
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        content = await file.read()
        temp_in.write(content)
        temp_in.close()
    except Exception as e:
        logger.error(f"Failed to save temp video: {e}")
        raise HTTPException(status_code=500, detail="Failed to save video on server.")

    temp_out_path = temp_in.name + "_out.mp4"

    def remove_files(paths: List[str]):
        for path in paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed temp file: {path}")
                except Exception as e:
                    logger.error(f"Error removing temp file {path}: {e}")

    try:
        # Open video capture
        cap = cv2.VideoCapture(temp_in.name)
        if not cap.isOpened():
            remove_files([temp_in.name])
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use mp4v as standard codec for generic compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

        yolo_model = get_model()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame
            results = yolo_model(frame, conf=conf)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        # Schedule cleanup of the temporary files
        background_tasks.add_task(remove_files, [temp_in.name, temp_out_path])

        return FileResponse(
            temp_out_path, 
            media_type="video/mp4", 
            filename="annotated_video.mp4"
        )

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        remove_files([temp_in.name, temp_out_path])
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# Mount static directory for Frontend files
# Note: StaticFiles should be mounted last or after specific endpoints
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
