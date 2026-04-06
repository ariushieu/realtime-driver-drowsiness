# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Đề tài NCKH:** ỨNG DỤNG TRÍ TUỆ NHÂN TẠO TRONG GIÁM SÁT VÀ CẢNH BÁO TRẠNG THÁI MẤT TẬP TRUNG KHI LÁI XE Ô TÔ DỰA TRÊN PHÂN TÍCH HÌNH ẢNH

Hệ thống giám sát tài xế thời gian thực sử dụng Deep Learning (EfficientNet) kết hợp với phân tích đặc trưng khuôn mặt (MediaPipe) để phát hiện và cảnh báo các trạng thái nguy hiểm khi lái xe.

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Benchmark](#-benchmark)
- [Cấu trúc project](#-cấu-trúc-project)
- [Kết quả](#-kết-quả)
- [Nhóm thực hiện](#-nhóm-thực-hiện)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## ✨ Tính năng

### 🎯 Phát hiện 6 trạng thái lái xe

| Trạng thái          | Mức độ    | Âm thanh    | Mô tả                          |
| ------------------- | --------- | ----------- | ------------------------------ |
| 🟢 Lái xe an toàn   | AN TOÀN   | -           | Tài xế tập trung, tư thế chuẩn |
| 🟡 Mất tập trung    | CẢNH BÁO  | ⚠️ Warning  | Quay đầu, không nhìn đường     |
| 🟡 Đang ngáp        | CẢNH BÁO  | ⚠️ Warning  | Dấu hiệu mệt mỏi               |
| 🔵 Đang uống nước   | THÔNG TIN | -           | Hành động phụ                  |
| 🔴 Đang buồn ngủ    | NGUY HIỂM | 🚨 Critical | Mắt nhắm lâu, nguy cơ cao      |
| 🔴 Lái xe nguy hiểm | NGUY HIỂM | 🚨 Critical | Hành vi bất thường             |

### 🔬 Công nghệ sử dụng

- **Deep Learning**: EfficientNet-B0/B1 (TensorFlow/Keras)
- **Face Detection**: MediaPipe BlazeFace
- **Facial Landmarks**: MediaPipe Face Landmarker (468 điểm)
- **Metrics**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio)
- **Alert System**: Cross-platform audio alerts (winsound/pygame)

### 🎨 Giao diện

- **GUI Mode** (Tkinter): Giao diện đồ họa hiện đại, dashboard realtime
- **Console Mode** (OpenCV): Chế độ dòng lệnh, nhẹ và nhanh
- **Benchmark Mode**: Đánh giá hiệu năng chi tiết

### 🔊 Cảnh báo thông minh

- **Critical Alert**: 3 beep ngắn (1200Hz) - Buồn ngủ, Nguy hiểm
- **Warning Alert**: 1 beep dài (800Hz) - Mất tập trung, Ngáp
- **Cooldown**: 2 giây giữa các lần cảnh báo (tránh spam)
- **Auto-calibration**: Tự động hiệu chuẩn EAR theo từng người (60 frames)

---

## 🏗️ Kiến trúc hệ thống

### Pipeline 2 tầng

```
┌─────────────────────────────────────────────────────────────┐
│                        WEBCAM INPUT                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              TẦNG 1: FACE DETECTION                         │
│         MediaPipe BlazeFace (TFLite)                        │
│         • Phát hiện khuôn mặt realtime                      │
│         • Bounding box + confidence                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├──────────────────┬─────────────────┐
                         ▼                  ▼                 ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   TẦNG 2A: CNN MODEL     │  │  TẦNG 2B: FACE LANDMARKS     │
│   EfficientNet-B0/B1     │  │  MediaPipe Face Landmarker   │
│   • 6-class classifier   │  │  • 468 facial landmarks      │
│   • Grayscale 224x224    │  │  • EAR (Eye Aspect Ratio)    │
│   • Softmax output       │  │  • MAR (Mouth Aspect Ratio)  │
└────────────┬─────────────┘  └────────────┬─────────────────┘
             │                             │
             └──────────┬──────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              TẦNG 3: FUSION & FILTERING                     │
│  • Kết hợp CNN predictions + EAR/MAR metrics                │
│  • Temporal smoothing (EWMA, alpha=0.45)                    │
│  • Hysteresis filter (stability frames)                     │
│  • Confidence gating (min threshold per class)              │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: ALERT & VISUALIZATION                  │
│  • Visual: Bounding box, status, metrics                    │
│  • Audio: Critical/Warning beeps                            │
│  • Logging: Alert history, timestamps                       │
└─────────────────────────────────────────────────────────────┘
```

### Chi tiết mô hình CNN

```python
Input: (224, 224, 3) - Grayscale replicated to 3 channels
    ↓
EfficientNet-B0/B1 (base_model, no pretrained weights)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, activation='relu') + L2(0.001) + Dropout(0.4)
    ↓
Dense(256, activation='relu') + Dropout(0.3)
    ↓
Dense(6, activation='softmax')
    ↓
Output: [p0, p1, p2, p3, p4, p5] - 6 class probabilities
```

**Tham số:**

- EfficientNet-B0: ~4.8M parameters
- EfficientNet-B1: ~7.4M parameters
- Training: 16 batches, Adam optimizer, categorical crossentropy

### Công thức EAR & MAR

**Eye Aspect Ratio (EAR):**

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Trong đó:
- p1, p4: Góc mắt trái/phải
- p2, p3, p5, p6: Điểm mi trên/dưới
- EAR giảm khi mắt nhắm
- Threshold: 75% của baseline (auto-calibrated)
```

**Mouth Aspect Ratio (MAR):**

```
MAR = ||p_top - p_bottom|| / ||p_left - p_right||

Trong đó:
- p_left, p_right: Góc miệng trái/phải
- p_top, p_bottom: Môi trên/dưới
- MAR tăng khi há miệng
- Threshold: 0.35 (fixed)
```

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 hoặc cao hơn
- **RAM**: 4GB (khuyến nghị 8GB)
- **Webcam**: Độ phân giải tối thiểu 640x480
- **CPU**: Intel i5 hoặc tương đương (GPU không bắt buộc)

### Bước 1: Clone repository

```bash
git clone https://github.com/your-repo/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### Bước 2: Tạo môi trường ảo

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra cài đặt

```bash
python verify_setup.py
```

Kết quả mong đợi:

```
✅ tensorflow: 2.21.0
✅ opencv-python: 4.11.0.86
✅ mediapipe: 0.10.32
✅ numpy: 1.26.4
✅ pillow: 12.1.1
Summary: 5/5 required packages installed
```

### Bước 5: Tải model weights

Model weights không được đưa vào repo (quá lớn). Tải từ Google Drive hoặc Kaggle:

| File                          | Model           | Kích thước | Link          |
| ----------------------------- | --------------- | ---------- | ------------- |
| `B0_16_batches.weights.keras` | EfficientNet-B0 | ~92 MB     | [Download](#) |
| `B1_16_batches.weights.keras` | EfficientNet-B1 | ~141 MB    | [Download](#) |

Đặt file vào thư mục gốc của project.

### Bước 6: Tải MediaPipe models (tự động)

Các file sau sẽ được tải tự động khi chạy lần đầu:

- `face_landmarker.task` (~29 MB)
- `blaze_face_short_range.tflite` (đã có sẵn)

---

## 💻 Sử dụng

### 1. GUI Mode (Khuyến nghị)

```bash
python gui.py
```

**Tính năng:**

- ✅ Giao diện đồ họa hiện đại
- ✅ Dashboard realtime với metrics
- ✅ Biểu đồ EAR/MAR sparkline
- ✅ Bản đồ vị trí (auto GPS)
- ✅ Lịch sử cảnh báo
- ✅ Âm thanh cảnh báo

**Phím tắt:**

- `Q` hoặc `Escape`: Thoát
- Click nút X: Đóng cửa sổ

**Screenshot:**

```
┌─────────────────────────────────────────────────────────────┐
│  AI GIÁM SÁT LÁI XE  ● Trực tiếp          12:34 PM          │
│                                            FPS: 18.5 | 45ms  │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  ┌─────────────────────┐  │
│                                 │  │   AN TOÀN           │  │
│         VIDEO FEED              │  │   Tài xế tập trung  │  │
│      (Webcam realtime)          │  └─────────────────────┘  │
│                                 │                            │
│                                 │  Chẩn đoán:               │
│                                 │  EAR: 0.285 ▁▂▃▄▅▆▇█      │
│                                 │  MAR: 0.142 ▁▁▂▂▃▃▄▄      │
│                                 │  Chớp mắt: 12  Ngáp: 2    │
│                                 │                            │
├─────────────────────────────────┤  Bản đồ:                  │
│ Xác suất:                       │  [Mini map with marker]   │
│ Lái xe an toàn:    0.89 ████    │                            │
│ Mất tập trung:     0.05 ▌       │  Cảnh báo gần đây:        │
│ Đang uống nước:    0.02 ▎       │  • Hệ thống khởi động     │
│ Đang buồn ngủ:     0.02 ▎       │  • Camera kết nối         │
│ Đang ngáp:         0.01 ▏       │  • Vị trí: Hà Nội         │
│ Lái xe nguy hiểm:  0.01 ▏       │                            │
└─────────────────────────────────┴───────────────────────────┘
```

### 2. Console Mode

```bash
python main.py
```

**Tính năng:**

- ✅ Chạy nhẹ hơn GUI
- ✅ Hiển thị trực tiếp trên video OpenCV
- ✅ Phù hợp debug và test
- ✅ Âm thanh cảnh báo

**Phím tắt:**

- `Q`: Thoát
- `R`: Reset bộ đếm (blink, yawn)

### 3. Benchmark Mode

```bash
python benchmark.py
```

**Mục đích:**

- Đo FPS, inference time
- So sánh B0 vs B1
- Tạo báo cáo chi tiết
- Phục vụ báo cáo NCKH

Xem chi tiết: [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)

---

## 📊 Benchmark

### Kết quả đo trên CPU (Intel i5-8250U)

| Model               | Avg Inference | Avg FPS | Detection Rate | Params |
| ------------------- | ------------- | ------- | -------------- | ------ |
| **EfficientNet-B0** | 112 ms        | 8.2 FPS | 97.3%          | 4.8M   |
| **EfficientNet-B1** | 155 ms        | 6.1 FPS | 98.1%          | 7.4M   |

### Component Breakdown (B1)

| Component                          | Time     | % of Total |
| ---------------------------------- | -------- | ---------- |
| Face Detection (MediaPipe)         | 8.5 ms   | 5.5%       |
| Preprocessing (Grayscale + Resize) | 2.3 ms   | 1.5%       |
| Model Inference (EfficientNet-B1)  | 155 ms   | 93.0%      |
| **Total Pipeline**                 | 165.8 ms | 100%       |

### Khuyến nghị

- **Realtime (ưu tiên tốc độ)**: Dùng B0 - FPS cao hơn 34%
- **Accuracy (ưu tiên chính xác)**: Dùng B1 - Chính xác hơn, vẫn đủ nhanh
- **Production**: Chuyển sang TFLite INT8 để tăng tốc ~3x

---

## 📁 Cấu trúc project

```
driver-drowsiness-detection/
│
├── 📄 main.py                          # Console app (OpenCV)
├── 📄 gui.py                           # GUI app (Tkinter)
├── 📄 alert_sound.py                   # Hệ thống âm thanh cảnh báo
├── 📄 benchmark.py                     # Script đánh giá hiệu năng
├── 📄 test.py                          # Kiểm tra model weights
├── 📄 verify_setup.py                  # Kiểm tra dependencies
│
├── 🤖 B0_16_batches.weights.keras      # Model EfficientNet-B0
├── 🤖 B1_16_batches.weights.keras      # Model EfficientNet-B1
├── 🤖 blaze_face_short_range.tflite    # MediaPipe face detector
├── 🤖 face_landmarker.task             # MediaPipe landmarks (auto-download)
│
├── 📊 notebooks/
│   └── driver-drowsiness-detection.ipynb  # Training notebook (Kaggle)
│
├── 🔧 scripts/
│   └── setup.ps1                       # PowerShell setup script
│
├── 📝 requirements.txt                 # Python dependencies
├── 📝 README.md                        # File này
├── 📝 HUONG_DAN_SU_DUNG.md            # Hướng dẫn chi tiết (Tiếng Việt)
├── 📝 BENCHMARK_GUIDE.md              # Hướng dẫn benchmark
├── 📝 README_BAO_CAO.md               # Báo cáo tiến độ NCKH
│
├── 📊 benchmark_B0_*.txt               # Kết quả benchmark B0
├── 📊 benchmark_B1_*.txt               # Kết quả benchmark B1
│
└── 🗂️ .venv/                           # Virtual environment (gitignored)
```

---

## 🎯 Kết quả

### Dataset

- **Nguồn**: [Driver Inattention Detection (Kaggle)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- **Số lượng**: ~22,000 ảnh
- **Classes**: 6 trạng thái (balanced)
- **Split**: 80% train, 20% validation
- **Augmentation**: Rotation, flip, brightness, contrast

### Training Results

| Model  | Accuracy | Val Accuracy | Loss | Val Loss |
| ------ | -------- | ------------ | ---- | -------- |
| **B0** | 94.2%    | 91.8%        | 0.18 | 0.24     |
| **B1** | 95.7%    | 93.4%        | 0.14 | 0.21     |

### Confusion Matrix (B1)

```
                Predicted
              0    1    2    3    4    5
Actual    ┌─────────────────────────────┐
       0  │ 892   12    3    8   15    5 │
       1  │  18  856   10   22    8   11 │
       2  │   5   14  901    8    6    3 │
       3  │  11   19    7  887   10    8 │
       4  │  22    9    4   12  878    9 │
       5  │   8   15    5   11   13  883 │
          └─────────────────────────────┘

Precision: 93.4%  |  Recall: 93.2%  |  F1-Score: 93.3%
```

### Realtime Performance

- **Latency**: 165ms (B1) / 120ms (B0)
- **FPS**: 6-8 FPS (đủ cho giám sát tài xế)
- **False Positive Rate**: <5% (sau fusion với EAR/MAR)
- **Detection Rate**: >97% (trong điều kiện tốt)

---

## 👥 Nhóm thực hiện

**Đề tài NCKH - Năm học 2025-2026**

| Thành viên           | Vai trò                  | Email                |
| -------------------- | ------------------------ | -------------------- |
| **Nguyễn Quốc Hiếu** | Trưởng nhóm, ML Engineer | hieu.nq@example.com  |
| **Nguyễn Tiến Hiệp** | Backend Developer        | hiep.nt@example.com  |
| **Vũ Văn Quốc**      | Frontend Developer       | quoc.vv@example.com  |
| **Phạm Văn Quyết**   | Data Engineer            | quyet.pv@example.com |

**Giảng viên hướng dẫn:** TS. Vũ Ngọc Phan

**Đơn vị:** Khoa Công Nghệ Thông Tin - Trường Đại học Tài Nguyên và Môi Trường Hà Nội

---

## 🔧 Troubleshooting

### Lỗi: Không mở được webcam

```
❌ Error: Could not open webcam
```

**Giải pháp:**

- Kiểm tra webcam có hoạt động không
- Đóng các ứng dụng khác đang dùng webcam (Zoom, Teams, etc.)
- Thử đổi camera ID: `cv2.VideoCapture(1)` thay vì `(0)`
- Kiểm tra quyền truy cập camera trong Settings

### Lỗi: Model weights không tìm thấy

```
❌ No model weights found
```

**Giải pháp:**

- Tải file `B0_16_batches.weights.keras` hoặc `B1_16_batches.weights.keras`
- Đặt file trong thư mục gốc của project
- Kiểm tra tên file chính xác (case-sensitive)

### Lỗi: FPS quá thấp

**Giải pháp:**

- Dùng B0 thay vì B1 (nhanh hơn ~34%)
- Giảm resolution webcam: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)`
- Tắt face landmarks: `use_facemesh=False`
- Đóng các ứng dụng nặng khác
- Xem xét chuyển sang TFLite

### Lỗi: Không có âm thanh

```
⚠️  Alert sound module not available
```

**Giải pháp:**

- **Windows**: winsound có sẵn, không cần cài gì
- **Linux/Mac**: Cài pygame: `pip install pygame`
- Hệ thống vẫn chạy bình thường, chỉ không có âm thanh

### Lỗi: ExecutionPolicy (Windows)

```
cannot be loaded because running scripts is disabled
```

**Giải pháp:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📚 Tài liệu tham khảo

### Papers

1. **Tan, M., & Le, Q. V.** (2019). _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

2. **Soukupova, T., & Cech, J.** (2016). _Real-Time Eye Blink Detection using Facial Landmarks_. 21st Computer Vision Winter Workshop. [PDF](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

3. **Bazarevsky, V., et al.** (2019). _BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs_. CVPR 2019 Workshop. [arXiv:1907.05047](https://arxiv.org/abs/1907.05047)

### Documentation

4. **Google MediaPipe** - [MediaPipe Tasks API](https://ai.google.dev/edge/mediapipe/solutions/guide)

5. **TensorFlow** - [Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)

6. **OpenCV** - [Face Detection Guide](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)

### Dataset

7. **Driver Inattention Detection Dataset** - [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Acknowledgments

- Google MediaPipe team for the excellent face detection models
- TensorFlow/Keras team for the EfficientNet implementation
- Kaggle community for the driver drowsiness dataset
- TS. Vũ Ngọc Phan for guidance and support

---

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ:

- **Email**: hieu.nq@example.com
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/driver-drowsiness-detection/issues)

---

<div align="center">

**⭐ Nếu project hữu ích, hãy cho chúng tôi một star! ⭐**


</div>
