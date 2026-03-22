"""
Driver Drowsiness Detection V2 - Improved Version
Features:
- MediaPipe Face Detection (more robust than Haar Cascades)
- Eye Aspect Ratio (EAR) for drowsiness detection
- Mouth Aspect Ratio (MAR) for yawn detection
- Temporal smoothing
- Smart alert system
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
import os
import mediapipe as mp

# Disable GPU to force CPU usage (remove if you have GPU)
# tf.config.set_visible_devices([], 'GPU')

# Class names - tieng Viet (khong dau vi OpenCV chi ho tro ASCII)
class_names = {
    0: 'Lai xe nguy hiem',
    1: 'Mat tap trung',
    2: 'Dang uong nuoc',
    3: 'Lai xe an toan',
    4: 'Dang buon ngu',
    5: 'Dang ngap'
}

# Cau hinh canh bao
alert_config = {
    0: {'color': (0, 0, 255),   'level': 'NGUY HIEM', 'sound': True},
    1: {'color': (0, 165, 255), 'level': 'CANH BAO',  'sound': True},
    2: {'color': (0, 255, 255), 'level': 'THONG TIN', 'sound': False},
    3: {'color': (0, 255, 0),   'level': 'AN TOAN',   'sound': False},
    4: {'color': (0, 0, 255),   'level': 'NGUY HIEM', 'sound': True},
    5: {'color': (0, 165, 255), 'level': 'CANH BAO',  'sound': True}
}

class ImprovedDriverDetector:
    def __init__(self, model_path="B0_16_batches.weights.keras", 
                 use_mediapipe=True, use_facemesh=True):
        """
        Initialize improved detector
        
        Args:
            model_path: Path to trained model weights
            use_mediapipe: Use MediaPipe for face detection (vs Haar Cascades)
            use_facemesh: Use FaceMesh for landmark detection (eyes, mouth tracking)
        """
        self.model_path = model_path
        self.model = None
        self.use_mediapipe = use_mediapipe
        self.use_facemesh = use_facemesh
        
        # MediaPipe solutions
        self.mp_face_detection = None
        self.mp_face_mesh = None
        self.face_detection = None
        self.face_mesh = None
        
        # Haar Cascade (fallback)
        self.face_cascade = None
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.inference_times = deque(maxlen=10)
        self.prediction_history = deque(maxlen=10)
        
        # Drowsiness metrics
        self.ear_history = deque(maxlen=20)  # Eye Aspect Ratio history
        self.mar_history = deque(maxlen=20)  # Mouth Aspect Ratio history
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsy_frames = 0
        self.was_yawning = False  # track yawn events (rising edge only)
        
        # Thresholds - EAR se duoc tu dong calibrate khi khoi dong
        self.EAR_THRESHOLD = 0.20        # Gia tri mac dinh, se cap nhat sau calibration
        self.EAR_BASELINE = None          # EAR trung binh khi mat mo (calibrated)
        self.MAR_THRESHOLD = 0.35
        self.DROWSY_FRAMES_THRESHOLD = 20
        self.CONFIDENCE_THRESHOLD = 0.40  # Ha xuong 0.40 de nhan dang cac class khac

        # Calibration
        self.CALIBRATION_FRAMES = 60      # So frame de do baseline EAR
        self.calibration_ear_buffer = []  # Buffer luu EAR trong pha calibration
        self.is_calibrated = False
        
        # Alert system
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        
        print("🚀 Initializing Improved Driver Drowsiness Detection System...")
        self._load_model()
        self._load_detectors()
        
    def _load_model(self):
        """Load the trained model"""
        print("🔄 Loading EfficientNet-B0 model...")
        
        base_model = tf.keras.applications.EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )
        # Pre-build base_model to create all internal layer variables
        # (e.g. EfficientNet's Normalization layer mean/variance/count)
        base_model(tf.zeros((1, 224, 224, 3)))
        base_model.trainable = True
        
        self.model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        self.model(tf.zeros((1, 224, 224, 3)))
        
        if os.path.exists(self.model_path):
            # skip_mismatch=True handles Normalization layer variable name differences
            # between the Keras version used during training and the current version.
            # The Normalization preprocessing layer stats are non-critical for inference.
            self.model.load_weights(self.model_path, skip_mismatch=True)
            print("✅ Model weights loaded successfully!")
        else:
            raise FileNotFoundError(f"❌ Model file not found: {self.model_path}")
    
    def _load_detectors(self):
        """Load face detection methods using MediaPipe Tasks API (mediapipe >= 0.10)"""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if self.use_mediapipe:
            print("🔄 Loading MediaPipe Face Detection (Tasks API)...")
            base_options = mp_python.BaseOptions(
                model_asset_path='blaze_face_short_range.tflite'
            )
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5
            )
            self.face_detector_task = mp_vision.FaceDetector.create_from_options(options)
            print("✅ MediaPipe Face Detection loaded!")

        if self.use_facemesh:
            landmarker_model = 'face_landmarker.task'
            if not os.path.exists(landmarker_model):
                print("🔄 Downloading Face Landmarker model...")
                import urllib.request
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                )
                try:
                    urllib.request.urlretrieve(url, landmarker_model)
                    print("✅ Face Landmarker model downloaded!")
                except Exception as e:
                    print(f"⚠️  Could not download Face Landmarker: {e}")
                    print("   EAR/MAR metrics will be disabled.")
                    self.use_facemesh = False

            if self.use_facemesh and os.path.exists(landmarker_model):
                print("🔄 Loading MediaPipe Face Landmarker...")
                base_options = mp_python.BaseOptions(model_asset_path=landmarker_model)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    num_faces=1
                )
                self.face_landmarker_task = mp_vision.FaceLandmarker.create_from_options(options)
                print("✅ MediaPipe Face Landmarker loaded!")

        # Haar Cascade as fallback
        print("🔄 Loading Haar Cascade (fallback)...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Haar Cascade loaded!")
    
    def calibrate_ear(self, ear):
        """
        Thu thap EAR trong pha calibration de tinh nguong phu hop voi tung nguoi.
        Threshold = baseline * 0.75 (mat nham khi EAR giam 25% so voi luc mo mat).
        """
        if self.is_calibrated:
            return
        self.calibration_ear_buffer.append(ear)
        if len(self.calibration_ear_buffer) >= self.CALIBRATION_FRAMES:
            self.EAR_BASELINE = np.mean(self.calibration_ear_buffer)
            self.EAR_THRESHOLD = self.EAR_BASELINE * 0.75
            self.is_calibrated = True
            print(f"✅ Calibration xong! EAR baseline: {self.EAR_BASELINE:.3f}, "
                  f"nguong buon ngu: {self.EAR_THRESHOLD:.3f}")

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: List of 6 landmarks for one eye
        Returns:
            EAR value (float)
        """
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        mouth_landmarks: [left_corner(61), right_corner(291), upper_inner(13), lower_inner(14)]
        MAR = vertical_opening / horizontal_width
        """
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])  # 61 -> 291
        vertical   = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])  # 13 -> 14
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def extract_facial_metrics(self, frame, face_landmarks):
        """
        Extract EAR and MAR from face landmarks
        
        Args:
            frame: Input frame
            face_landmarks: MediaPipe face landmarks
        Returns:
            dict with 'ear', 'mar', 'is_drowsy', 'is_yawning'
        """
        if not face_landmarks:
            return None

        h, w = frame.shape[:2]
        # Tasks API returns a plain list of NormalizedLandmark directly
        landmarks = face_landmarks

        # MediaPipe Face Mesh landmark indices
        left_eye_indices  = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        # 4 diem mieng: goc trai(61), goc phai(291), moi tren trong(13), moi duoi trong(14)
        mouth_indices     = [61, 291, 13, 14]

        left_eye  = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                              for i in left_eye_indices])
        right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                              for i in right_eye_indices])
        mouth     = np.array([[landmarks[i].x * w, landmarks[i].y * h]
                              for i in mouth_indices])
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mar = self.calculate_mar(mouth)
        
        # Track history
        self.ear_history.append(avg_ear)
        self.mar_history.append(mar)

        # Cap nhat calibration truoc khi dung threshold
        self.calibrate_ear(avg_ear)

        # Determine states
        is_drowsy = avg_ear < self.EAR_THRESHOLD
        is_yawning = mar > self.MAR_THRESHOLD
        
        # Count consecutive drowsy frames
        if is_drowsy:
            self.drowsy_frames += 1
        else:
            if self.drowsy_frames > 0:
                self.blink_counter += 1
            self.drowsy_frames = 0
        
        # Count yawn events (rising edge: False -> True)
        if is_yawning and not self.was_yawning:
            self.yawn_counter += 1
        self.was_yawning = is_yawning
        
        return {
            'ear': avg_ear,
            'mar': mar,
            'is_drowsy': is_drowsy,
            'is_yawning': is_yawning,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth
        }
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe Tasks API"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_detector_task.detect(mp_image)

        h, w = frame.shape[:2]
        faces = []
        for detection in results.detections:
            bb = detection.bounding_box          # absolute pixel coords
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            fw = min(bb.width,  w - x)
            fh = min(bb.height, h - y)
            faces.append((x, y, fw, fh))

        return faces
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascades (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        return faces
    
    def detect_face_landmarks(self, frame):
        """Detect facial landmarks using MediaPipe Face Landmarker (Tasks API)"""
        if not hasattr(self, 'face_landmarker_task'):
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_landmarker_task.detect(mp_image)

        if results.face_landmarks:
            return results.face_landmarks[0]  # list of NormalizedLandmark
        return None
    
    def preprocess_face(self, face_img):
        """Preprocess face image - exact match with training pipeline"""
        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale and duplicate to 3 channels (matching training)
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        face_3channel = np.stack([face_gray, face_gray, face_gray], axis=-1)
        
        face_normalized = face_3channel.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def fuse_prediction(self, predicted_class, confidence, facial_metrics):
        """
        Ket hop ket qua model CNN voi chi so EAR/MAR.
        Nguyen tac: EAR/MAR chi dung de xac nhan hoac bac bo class buon ngu/ngap.
        Cac class khac (mat tap trung, uong nuoc, nguy hiem) de model tu quyet dinh.
        """
        # 1. Model khong du tin cay -> mac dinh an toan
        if confidence < self.CONFIDENCE_THRESHOLD:
            return 3, confidence  # SafeDriving

        if facial_metrics:
            ear_drowsy = facial_metrics['is_drowsy']
            mar_yawning = facial_metrics['is_yawning']

            # 2. Model bao buon ngu (4) nhung EAR khong xac nhan -> ha cap
            if predicted_class == 4 and not ear_drowsy:
                return 3, confidence  # SafeDriving

            # 3. Model bao ngap (5) nhung MAR khong xac nhan -> ha cap
            if predicted_class == 5 and not mar_yawning:
                return 3, confidence  # SafeDriving

            # 4. EAR thap lien tuc (> 5 frames) nhung model bao an toan/mat tap trung
            #    -> nang cap len buon ngu
            if self.drowsy_frames > 5 and predicted_class in (1, 3):
                return 4, confidence  # SleepyDriving

        # Cac class con lai (0-DangerousDriving, 1-Distracted, 2-Drinking, 3-Safe)
        # giu nguyen ket qua model neu confidence du cao
        return predicted_class, confidence

    def smooth_predictions(self, prediction):
        """Smooth predictions over time"""
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) < 3:
            return prediction
        
        recent_predictions = np.array(list(self.prediction_history))
        smoothed = np.mean(recent_predictions, axis=0)
        
        return smoothed
    
    def should_alert(self, predicted_class, facial_metrics=None):
        """Determine if alert should be triggered"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        # Critical: Eyes closed for too long
        if facial_metrics and facial_metrics['is_drowsy']:
            if self.drowsy_frames > self.DROWSY_FRAMES_THRESHOLD:
                self.last_alert_time = current_time
                return True
        
        # Check model prediction
        if alert_config[predicted_class]['sound']:
            self.last_alert_time = current_time
            return True
        
        return False
    
    def draw_facial_features(self, frame, facial_metrics):
        """Draw landmarks in Tech/Sci-fi style (Dots + Thin Lines)"""
        if not facial_metrics:
            return

        # Colors (BGR)
        # Cyan for normal state (rat noi tren nen toi/camera)
        color_normal = (255, 255, 0)   
        color_alert  = (0, 0, 255)     # Red
        color_warn   = (0, 140, 255)   # Orange

        # Determine colors based on state
        eye_color = color_alert if self.drowsy_frames > 3 else color_normal
        mouth_color = color_warn if facial_metrics['is_yawning'] else color_normal

        # --- DRAW EYES ---
        # Line thickness: 1 (elegant), Dot radius: 2
        for eye_name in ['left_eye', 'right_eye']:
            pts = facial_metrics[eye_name].astype(np.int32)
            
            # 1. Draw smooth contour
            cv2.polylines(frame, [pts], True, eye_color, 1, cv2.LINE_AA)
            
            # 2. Draw dots at keypoints (Tech look)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 2, eye_color, -1, cv2.LINE_AA)

        # --- DRAW MOUTH ---
        # Re-order points to form a diamond shape: Left->Top->Right->Bottom
        corners = facial_metrics['mouth'].astype(np.int32)
        # indices from extract: 0:Left, 1:Right, 2:Top, 3:Bottom
        mouth_poly = np.array([corners[0], corners[2], corners[1], corners[3]])
        
        # 1. Draw contour
        cv2.polylines(frame, [mouth_poly], True, mouth_color, 1, cv2.LINE_AA)
        
        # 2. Draw dots
        for pt in mouth_poly:
            cv2.circle(frame, tuple(pt), 2, mouth_color, -1, cv2.LINE_AA)
    
    def draw_ui(self, frame, faces, predictions_data, facial_metrics=None):
        """Draw enhanced UI"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw faces and predictions
        for i, ((x, y, fw, fh), pred_data) in enumerate(zip(faces, predictions_data)):
            predicted_class, confidence, all_probs = pred_data
            
            alert_info = alert_config[predicted_class]
            color = alert_info['color']
            level = alert_info['level']
            
            # Draw face bounding box
            thickness = 3 if level in ['CRITICAL', 'WARNING'] else 2
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, thickness)

            # Ve main label va cap do canh bao
            class_name = class_names[predicted_class]
            main_text = f"{class_name} ({confidence:.0%})"
            level_text = f"[{alert_info['level']}]"

            font_scale = 0.65
            text_thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(main_text, font, font_scale, text_thickness)
            (level_w, level_h), _ = cv2.getTextSize(level_text, font, 0.5, 1)

            bg_y = max(0, y - text_h - level_h - 15)
            cv2.rectangle(frame, (x, bg_y), (x + max(text_w, level_w) + 10, y), color, -1)

            cv2.putText(frame, main_text, (x+5, y-level_h-5), font, font_scale,
                       (255, 255, 255), text_thickness)
            cv2.putText(frame, level_text, (x+5, y-5), font, 0.5, (255, 255, 255), 1)

            # Probability bars (ben phai man hinh)
            bar_x = w - 230
            bar_y_start = 50
            bar_w = 100
            bar_h = 14
            cv2.putText(frame, "Xac suat:", (bar_x, bar_y_start - 12), font, 0.5,
                       (200, 200, 200), 1)
            for cls_id, prob in enumerate(all_probs):
                yp = bar_y_start + cls_id * (bar_h + 4)
                filled = int(bar_w * float(prob))
                bg_color = (50, 50, 50)
                fg_color = alert_config[cls_id]['color']
                if cls_id == predicted_class:
                    fg_color = color  # highlight selected
                cv2.rectangle(frame, (bar_x, yp), (bar_x + bar_w, yp + bar_h), bg_color, -1)
                cv2.rectangle(frame, (bar_x, yp), (bar_x + filled, yp + bar_h), fg_color, -1)
                label = f"{list(class_names.values())[cls_id][:10]}: {prob:.0%}"
                cv2.putText(frame, label, (bar_x + bar_w + 5, yp + bar_h - 1),
                           font, 0.38, (220, 220, 220), 1)
        
        # Hien thi chi so khuon mat
        if facial_metrics:
            metrics_y = 100

            # Hien thi trang thai calibration
            if not self.is_calibrated:
                remaining = self.CALIBRATION_FRAMES - len(self.calibration_ear_buffer)
                cal_text = f"Calibrating... con {remaining} frames"
                cv2.putText(frame, cal_text, (10, metrics_y - 25), font, 0.55,
                           (0, 255, 255), 2)

            ear_color = (0, 0, 255) if self.drowsy_frames > 3 else (0, 255, 0)
            cv2.putText(frame, f"Mat (EAR): {facial_metrics['ear']:.3f}",
                       (10, metrics_y), font, 0.6, ear_color, 2)

            mar_color = (0, 165, 255) if facial_metrics['is_yawning'] else (0, 255, 0)
            yawn_label = " [NGAP!]" if facial_metrics['is_yawning'] else ""
            cv2.putText(frame, f"Mieng (MAR): {facial_metrics['mar']:.3f}{yawn_label}",
                       (10, metrics_y + 30), font, 0.6, mar_color, 2)

            cv2.putText(frame, f"Nhay mat: {self.blink_counter} lan",
                       (10, metrics_y + 60), font, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Ngap: {self.yawn_counter} lan",
                       (10, metrics_y + 90), font, 0.6, (255, 255, 255), 2)

            # Canh bao buon ngu
            if self.drowsy_frames > self.DROWSY_FRAMES_THRESHOLD:
                warning_text = "!!! DANG BUON NGU !!!"
                (tw, th), _ = cv2.getTextSize(warning_text, font, 1.0, 3)
                cv2.rectangle(frame, ((w - tw) // 2 - 10, h // 2 - th - 10),
                             ((w + tw) // 2 + 10, h // 2 + 10), (0, 0, 200), -1)
                cv2.putText(frame, warning_text, ((w - tw) // 2, h // 2),
                           font, 1.0, (255, 255, 255), 3)

        # Thong tin hieu suat
        if self.fps_counter:
            cv2.putText(frame, f"FPS: {np.mean(self.fps_counter):.1f}",
                       (10, 30), font, 0.7, (0, 255, 0), 2)

        if self.inference_times:
            cv2.putText(frame, f"Xu ly: {np.mean(self.inference_times)*1000:.0f}ms",
                       (10, 60), font, 0.7, (0, 255, 0), 2)

        method = "MediaPipe" if self.use_mediapipe else "Haar"
        cv2.putText(frame, f"Nhan dang: {method}", (w - 220, 30), font, 0.6,
                   (255, 255, 255), 2)
    
    def run(self):
        """Main application loop"""
        print("📹 Starting webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ System ready! Press 'q' to quit")
        print(f"   Face Detection: {'MediaPipe' if self.use_mediapipe else 'Haar Cascades'}")
        print(f"   Facial Metrics: {'Enabled' if self.use_facemesh else 'Disabled'}")
        print(f"   Dang calibrate EAR trong {self.CALIBRATION_FRAMES} frames dau...")
        print(f"   (Nhin thang vao camera, giu mat mo binh thuong)")
        
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                if self.use_mediapipe:
                    faces = self.detect_faces_mediapipe(frame)
                else:
                    faces = self.detect_faces_haar(frame)
                
                # Extract facial landmarks (if enabled)
                facial_metrics = None
                if self.use_facemesh and len(faces) > 0:
                    face_landmarks = self.detect_face_landmarks(frame)
                    if face_landmarks:
                        facial_metrics = self.extract_facial_metrics(frame, face_landmarks)
                        self.draw_facial_features(frame, facial_metrics)
                
                predictions_data = []
                
                # Process each face
                for (x, y, w, h) in faces:
                    if w < 50 or h < 50:
                        continue
                    
                    face = frame[y:y+h, x:x+w]
                    
                    # Preprocess and predict
                    face_input = self.preprocess_face(face)
                    
                    inference_start = time.time()
                    predictions = self.model.predict(face_input, verbose=0)
                    inference_time = time.time() - inference_start
                    self.inference_times.append(inference_time)
                    
                    smoothed_predictions = self.smooth_predictions(predictions[0])
                    
                    predicted_class = np.argmax(smoothed_predictions)
                    confidence = float(smoothed_predictions[predicted_class])

                    # Ket hop EAR/MAR voi model de giam false positive
                    predicted_class, confidence = self.fuse_prediction(
                        predicted_class, confidence, facial_metrics
                    )
                    
                    predictions_data.append((predicted_class, confidence, smoothed_predictions))
                    
                    # Check for alerts
                    if self.should_alert(predicted_class, facial_metrics):
                        print(f"🚨 ALERT: {class_names[predicted_class]} detected!")
                        if facial_metrics and facial_metrics['is_drowsy']:
                            print(f"   Eyes closed for {self.drowsy_frames} frames!")
                
                # Draw UI
                self.draw_ui(frame, faces, predictions_data, facial_metrics)
                
                # Calculate FPS
                frame_time = time.time() - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_counter.append(fps)
                
                cv2.imshow('Driver Drowsiness Detection V2', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset counters
                    self.blink_counter = 0
                    self.yawn_counter = 0
                    self.drowsy_frames = 0
                    print("🔄 Counters reset")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Cleanup MediaPipe Tasks
            if hasattr(self, 'face_detector_task'):
                self.face_detector_task.close()
            if hasattr(self, 'face_landmarker_task'):
                self.face_landmarker_task.close()
            
            print(f"\n📈 Final Statistics:")
            if self.fps_counter:
                print(f"   Average FPS: {np.mean(self.fps_counter):.1f}")
            if self.inference_times:
                print(f"   Average Inference: {np.mean(self.inference_times)*1000:.0f}ms")
            print(f"   Total Frames: {frame_count}")
            print(f"   Blinks: {self.blink_counter}")
            print(f"   Yawns: {self.yawn_counter}")
            print("👋 Application closed!")


def main():
    print("=" * 70)
    print("🚗 Driver Drowsiness Detection System V2 (Improved)")
    print("   Features: MediaPipe, EAR/MAR metrics, Enhanced alerts")
    print("=" * 70)
    
    try:
        detector = ImprovedDriverDetector(
            use_mediapipe=True,   # Set False to use Haar Cascades
            use_facemesh=True     # Set False to disable facial metrics
        )
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
