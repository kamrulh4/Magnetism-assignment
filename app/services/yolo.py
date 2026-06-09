from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from app.core.config import get_model, logger

def detect_objects_in_image(img: np.ndarray, conf: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Performs object detection on a numpy image array using YOLOv8n.
    Returns:
        detections: List of dictionaries containing bounding box, confidence, and label.
        annotated_img: OpenCV image (numpy array) containing the drawn bounding boxes.
    """
    yolo_model = get_model()
    results = yolo_model(img, conf=conf)
    
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
        
    annotated_img = result.plot()
    return detections, annotated_img

def process_video(input_path: str, output_path: str, conf: float) -> None:
    """
    Processes a video frame-by-frame, applying YOLOv8n inference and drawing bounding boxes,
    then writes the processed frames to output_path.
    """
    logger.info(f"Processing video: {input_path} -> {output_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v as standard codec for generic compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on frame
            detections_result = get_model()(frame, conf=conf)
            annotated_frame = detections_result[0].plot()
            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()
    logger.info("Finished processing video successfully.")
