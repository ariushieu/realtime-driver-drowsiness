# Driver Drowsiness Detection

Real-time driver drowsiness detection using EfficientNet + MediaPipe, built as a university research project (NCKH).

## How it works

Two-layer detection pipeline:

1. **EfficientNet-B0/B1** classifies the driver face into 6 states
2. **MediaPipe FaceLandmarker** computes EAR (eye) and MAR (mouth) metrics
3. **Fusion logic** combines both layers to reduce false positives

| State             | Label     | Level   |
| ----------------- | --------- | ------- |
| Safe driving      | AN TOAN   | Normal  |
| Distracted        | CANH BAO  | Warning |
| Yawning           | CANH BAO  | Warning |
| Drinking water    | THONG TIN | Info    |
| Drowsy            | NGUY HIEM | Danger  |
| Dangerous driving | NGUY HIEM | Danger  |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python verify_setup.py
```

## Run

```powershell
python main.py
```

On first run, `face_landmarker.task` (~29 MB) will be downloaded automatically.  
Keep your eyes open and look straight ahead for ~8 seconds during EAR calibration.

**Shortcuts:** `Q` quit · `R` reset counters

## Model weights

Not included in the repo (too large). Download from Kaggle and place in the project root:

| File                          | Model           | Size    |
| ----------------------------- | --------------- | ------- |
| `B0_16_batches.weights.keras` | EfficientNet-B0 | ~92 MB  |
| `B1_16_batches.weights.keras` | EfficientNet-B1 | ~141 MB |

Dataset: [Driver Inattention Detection](https://www.kaggle.com/code/fissalalsharef/driver-drowsiness-detection)

## Project structure

```
driver-drowsiness-detection/
├── driver-drowsiness-detection.ipynb  # Training notebook (Kaggle)
├── main.py                            # Real-time detection (MediaPipe + EAR/MAR)
├── blaze_face_short_range.tflite      # BlazeFace model
├── requirements.txt
├── verify_setup.py
└── README.md
```

## Architecture

```
Webcam frame
     |
     v
MediaPipe BlazeFace --> Crop face region
     |                       |
     v                       v
FaceLandmarker         EfficientNet-B0/B1
     |                       |
     v                       v
EAR (eyes)           6-class softmax
MAR (mouth)                  |
     |                       |
     +-------- Fusion --------+
                   |
                   v
          Final result + Alert
```

**Model:**

```
EfficientNetB0 (no pretrained weights)
GlobalAveragePooling2D
BatchNormalization
Dense(512, relu) + Dropout(0.4)
Dense(256, relu) + Dropout(0.3)
Dense(6, softmax)   -- ~4.8M params (B0), ~7.4M params (B1)
```

**Eye Aspect Ratio (EAR)** — detects closed eyes. Threshold is auto-calibrated per user over the first 60 frames.

**Mouth Aspect Ratio (MAR)** — detects yawning. MAR > 0.35 triggers yawn detection.

## Performance (CPU, no GPU)

| Model           | Inference | FPS    |
| --------------- | --------- | ------ |
| EfficientNet-B0 | ~112 ms   | ~8 FPS |
| EfficientNet-B1 | ~155 ms   | ~6 FPS |

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`** — activate venv first:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**`Layer 'normalization' expected 3 variables`** — already fixed (pre-build + `skip_mismatch=True`).

**`module 'mediapipe' has no attribute 'solutions'`** — already fixed (MediaPipe Tasks API).

**Webcam won't open** — try `cv2.VideoCapture(1)` or check camera permissions.

**ExecutionPolicy error:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Roadmap

- [ ] Convert model to TFLite INT8 (~3x faster inference)
- [ ] Audio alert on drowsiness
- [ ] LSTM temporal context
- [ ] Edge deployment (Raspberry Pi / Jetson Nano)

## References

1. Tan & Le — _EfficientNet: Rethinking Model Scaling for CNNs_ (ICML 2019)
2. Soukupova & Cech — _Real-Time Eye Blink Detection using Facial Landmarks_ (2016)
3. Google — [MediaPipe Tasks API](https://ai.google.dev/edge/mediapipe/solutions/guide)
4. [Driver Inattention Detection Dataset](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
