import os
import io
import cv2
import numpy as np
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "YOLOv8n"

def test_detect_image_invalid_type():
    # Test sending text instead of image
    response = client.post(
        "/detect-image/",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"conf": 0.25}
    )
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_detect_image_success():
    # Create a synthetic image (solid blue 200x200 pixels)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255 # Fill blue channel
    
    # Encode as JPEG bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    response = client.post(
        "/detect-image/",
        files={"file": ("test_image.jpg", img_bytes, "image/jpeg")},
        data={"conf": 0.25}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "annotated_image" in data
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")
    # For a solid blue image, we expect 0 detections
    assert isinstance(data["detections"], list)

def test_detect_video_invalid_type():
    # Test sending image in video endpoint
    response = client.post(
        "/detect-video/",
        files={"file": ("test.jpg", b"fake", "image/jpeg")},
        data={"conf": 0.25}
    )
    assert response.status_code == 400
    assert "must be a video" in response.json()["detail"]

def test_detect_video_success():
    # Create a small synthetic video with 3 frames (100x100 size)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video_path = temp_video.name
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, 10.0, (100, 100))
    
    # Write 3 synthetic frames
    for _ in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # Read the video bytes
    with open(temp_video_path, "rb") as f:
        video_bytes = f.read()
        
    # Clean up local video file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    response = client.post(
        "/detect-video/",
        files={"file": ("test_video.mp4", video_bytes, "video/mp4")},
        data={"conf": 0.25}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    # Ensure some video data was returned
    assert len(response.content) > 0
