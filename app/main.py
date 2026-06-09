import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Triggers configuration loading, folders creation, and PyTorch patch
from app.core.config import get_model, STATIC_DIR, logger
from app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize YOLOv8 model loading during startup lifespan
    get_model()
    yield

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="A modular FastAPI backend serving YOLOv8n object detection on images and videos",
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

# Register the routes router
app.include_router(api_router)

# Mount static folder (located at the project root)
logger.info(f"Mounting static files from: {STATIC_DIR}")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
