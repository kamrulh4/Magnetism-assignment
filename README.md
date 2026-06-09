# VisionAI - YOLOv8 Object Detection Web Application

A premium, production-ready AI-integrated web application that performs object detection on images and videos using a pretrained **YOLOv8n** model. Built with a high-performance **FastAPI** backend and a custom, responsive **Vanilla HTML/CSS/JS** frontend featuring glassmorphic designs, interactive telemetry tables, and media download support.

---

## 🌟 Key Features

* **Real-time Image Detection**: Upload any image (JPEG, PNG) and view the annotated output with bounding boxes, labels, and confidence levels.
* **Confidence Threshold Control**: Adjust a slider (0.05 to 1.0) to filter detections dynamically.
* **Interactive Telemetry Dashboard**: Get count stats and a detailed list of all detected objects, complete with confidence pill indicators and coordinate details.
* **Annotated Media Download**: One-click download of the annotated images or processed videos directly from the browser.
* **Video Inference (Bonus)**: Upload a video (MP4, WebM), process it frame-by-frame on the backend, and play back or download the processed video.
* **Docker Support (Bonus)**: Containerized setup using Docker and Docker Compose with weights pre-packaged to ensure rapid deployment.
* **Unit Test Suite**: End-to-end endpoint verification covering error states and custom multi-part file uploads.

---

## 🏗️ Architecture and Design Flow

```
+--------------------------------------------------------------+
|                          Web UI                              |
|   (HTML5, Modern CSS Glassmorphism, Vanilla Javascript)     |
+------------------------------+-------------------------------+
                               |
                   HTTP POST   |   Multipart Form Upload
                   (File, Conf)|
                               v
+--------------------------------------------------------------+
|                      FastAPI Backend                         |
|   - CORS Middleware enabled                                  |
|   - Lazy-loaded YOLOv8n model cached in memory               |
|   - Temporary file background cleanup task queue            |
+------------------------------+-------------------------------+
                               |
                     Inference |   PyTorch/Ultralytics
                               v
+--------------------------------------------------------------+
|                        YOLOv8n Model                         |
|   - Pre-downloaded weights (yolov8n.pt)                      |
+--------------------------------------------------------------+
```

---

## ⚙️ Setup and Installation

### Option 1: Standard Local Setup (Python virtual environment)

Ensure you have Python 3.9+ installed on your system.

1. **Clone or navigate to the repository directory**:
   ```bash
   cd Magnetism-assignment
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the FastAPI application**:
   ```bash
   python3 main.py
   ```
   The application will start on **`http://127.0.0.1:8000`**.

---

### Option 2: Docker Setup (Recommended for rapid testing)

You only need Docker and Docker Compose installed.

1. **Run the following command in the root directory**:
   ```bash
   docker compose up --build
   ```
   This command compiles the environment, pre-downloads the model weights during the build phase (saving runtime downloading delay), and serves the app.

2. **Open your browser and navigate to**:
   `http://localhost:8000`

---

## 🧪 Running Unit Tests

We have written comprehensive unit tests to ensure API stability and boundary checks.

Run tests using `pytest` inside your virtual environment:
```bash
pytest test_main.py -v
```

---

## 🧠 Design Decisions & Key Assumptions

1. **FastAPI vs. Django**:
   FastAPI was selected for its ultra-lightweight profile, built-in asynchronous framework support (which is crucial for video frame processing), auto-generated Swagger docs, and low-latency response times.
2. **Vanilla HTML/CSS/JS vs. React/Next.js**:
   While React is great, vanilla frontend assets (HTML, CSS, and JS) were selected to avoid bloating the repository with complex node_modules, webpack builds, or Babel compilers. Modern vanilla CSS features such as variables, flexbox/grid, and glassmorphic blurs (`backdrop-filter`) allow for high-end styling with zero build overhead.
3. **OpenCV Headless (`opencv-python-headless`)**:
   We used the headless version of OpenCV to prevent image build errors in Docker and server environments where GUI frameworks (like X11/Qt) are absent.
4. **Weights Pre-packaging**:
   We download the YOLO weights during the Docker image compilation phase instead of container runtime. This ensures the service works immediately upon startup even in network-constrained environments.
5. **Video Processing Memory Optimization**:
   Video files are streamed through OpenCV frame-by-frame and written to a temporary output file instead of reading all frames into RAM. This ensures memory usage remains low even for larger video uploads. A background thread automatically garbage-collects these temp files after the download response completes.
