# Baseball Detection

Training and exporting a custom baseball detector built on a YOLOv8 P2-heavy backbone. The workflow downloads a Roboflow dataset, applies optional motion-blur augmentation, trains with Ultralytics, and exports to ONNX/TFLite/CoreML.

## What's Inside
- **`model_test.py`** – **Main inference script**: Implements the full ball tracking pipeline (Pose Analysis + Ball Detection).
- `baseball_detect.py` – End-to-end training script: download Roboflow dataset, optional cleanup/augmentation, train YOLO with a custom `yolov8n-p2-new.yaml`.
- `yolov8n-p2-new.yaml` – Custom model with extra P2 focus (P2–P4 head, P5 removed).
- `export.py` – Export trained weights to TFLite (float16).
- `export_coreML.py` – Convert an ONNX export to a CoreML `.mlpackage`.
- `requirements*.txt` – CUDA 12.1 friendly dependency set; use `requirements.txt` by default.

## Inference / Tracking Pipeline (`model_test.py`)
The system uses a two-stage pipeline to detect and track the baseball:

### 1. Pose Analysis (Initialization)
- Uses `yolov8n-pose.pt` to detect the pitcher's **Right Wrist** (COCO Keypoint 10).
- Determines the initial **Region of Interest (ROI)** (480x320) centered around the wrist.
- Sets the tracking direction based on the hand's position relative to the frame center (e.g., if hand is on left, ball is expected to move right).

### 2. Ball Detection & Tracking
- **Search Mode**: Scans the area around the wrist to find the ball release.
- **Track Mode (Sticky)**:
  - Once the ball is detected and moves away from the hand (>15px), the system locks into **Track Mode**.
  - ROI is dynamically centered on the predicted ball position (offset 50px in the flight direction).
  - **Fixed Extrapolation**: If detection is lost (e.g., motion blur), the ROI continues to move 100px/frame in the tracking direction to catch the ball again.
- **Model**: Uses a custom TFLite model (e.g., `yolov8n_p2new_mblur_40_db_float16.tflite`) optimized for small objects.

## Usage
### Run Inference
```bash
python model_test.py -v video_name
```
- Looks for videos in `videos/` folder.
- Outputs annotated frames to `results/`.

## Setup
1) Python 3.10 recommended (repo uses `venv310/` locally).
2) Create and activate a venv, then install deps:
   ```bash
   python -m venv venv
   venv\\Scripts\\activate      # PowerShell on Windows
   pip install -r requirements.txt
   ```
   For a smaller install without TensorFlow tooling, switch to `requirements-cu121.txt` if preferred.
3) Ensure NVIDIA drivers match CUDA 12.1 for the pinned PyTorch/torchvision builds.

## Data
- Dataset is pulled from Roboflow project `mickshelbytsai/pitch-tracking-sgse6`, version 2, format `yolov8`.
- Create a `.env` file in the project root and add your Roboflow API key:
  ```env
  ROBOFLOW_API_KEY=your_key_here
  ```
  The script `baseball_detect.py` uses `python-dotenv` to load this key automatically.
- Downloaded data lives under `datasets/` (gitignored).

## Training
Run the main script:
```bash
python baseball_detect.py
```
Key options inside `baseball_detect.py`:
- `augment_dataset(dataset.location, augment_ratio=0.25)` applies motion-blur + brightness/contrast adjustments to a random subset of training images while cloning labels.
- `remove_excess_negatives(...)` can drop a fraction of empty-label samples; currently commented out.
- Training defaults: `epochs=200`, `patience=30`, `imgsz=1280`, `batch=4`, `device=0`, `workers=0` (Windows-safe).
- The run is named via `name = "yolov8n_p2new_mblur_25_pt"` and saved under `runs/detect/<name>/`.

## Export
- TFLite (float16): adjust the weights path in `export.py`, then `python export.py` to write to `runs/detect/.../weights/best_float16.tflite` (default Ultralytics pathing).
- ONNX/CoreML: after training, export ONNX from Ultralytics (see commented block in `baseball_detect.py`), place it in `export/`, then run:
  ```bash
  python export_coreML.py
  ```
  to produce `export/<model>.mlpackage`.

## Tips
- GPU OOM: lower `imgsz` or `batch` in `baseball_detect.py`.
- Slow dataloading on Windows: `workers=0` is intentional; raise cautiously.
- To swap models, point `YOLO("./yolov8n-p2-new.yaml")` at `yolov11n-p2.yaml` or `yolov11s-p2.yaml` variants already in the repo.
