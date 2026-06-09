import os
import cv2
import numpy as np
import base64
import tempfile
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from app.core.config import logger
from app.services.yolo import detect_objects_in_image, process_video

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "healthy", "model": "YOLOv8n"}

@router.post("/detect-image/")
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

        # Run inference using service layer
        detections, annotated_img = detect_objects_in_image(img, conf=conf)
        
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

@router.post("/detect-video/")
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
        # Check input open state
        cap = cv2.VideoCapture(temp_in.name)
        if not cap.isOpened():
            remove_files([temp_in.name])
            raise HTTPException(status_code=400, detail="Could not open video file.")
        cap.release()

        # Run process video service layer
        process_video(temp_in.name, temp_out_path, conf=conf)
        
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
