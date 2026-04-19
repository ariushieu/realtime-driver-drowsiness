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
import threading
import mediapipe as mp

# Sound alert: Windows built-in (no external files needed)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
    print("⚠️  winsound not available (non-Windows). Sound alerts disabled.")

# Disable GPU to force CPU usage (remove if you have GPU)
# tf.config.set_visible_devices([], 'GPU')

# Class names 
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
    def __init__(self, model_path=None,
                 use_mediapipe=True, use_facemesh=True):
        """
        Initialize improved detector

        Args:
            model_path: Path to trained model weights (auto-detect if None)
            use_mediapipe: Use MediaPipe for face detection (vs Haar Cascades)
            use_facemesh: Use FaceMesh for landmark detection (eyes, mouth tracking)
        """
        # Auto-detect model weights: prefer B0, fallback to B1
        if model_path is None:
            if os.path.exists("B0_16_batches.weights.keras"):
                model_path = "B0_16_batches.weights.keras"
            elif os.path.exists("B1_16_batches.weights.keras"):
                model_path = "B1_16_batches.weights.keras"
            else:
                raise FileNotFoundError(
                    "No model weights found. Expected B0_16_batches.weights.keras "
                    "or B1_16_batches.weights.keras"
                )
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
        self.prediction_history = deque(maxlen=8)  # tang buffer de smooth tot hon
        
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
        
        # IMPROVED: Temporal smoothing cho predictions
        self.class_vote_history = deque(maxlen=5)  # Luu 5 predictions gan nhat
        self.last_stable_class = 3  # Mac dinh Safe

        # Calibration
        self.CALIBRATION_FRAMES = 60      # So frame de do baseline EAR
        self.calibration_ear_buffer = []  # Buffer luu EAR trong pha calibration
        self.is_calibrated = False
        
        # Stability filter: class phai thang lien tuc N frame moi duoc hien thi
        self.stable_class = 3          # class dang hien thi (mac dinh Safe)
        self.stable_confidence = 0.0
        self.candidate_class = 3       # class dang "cho" du frame
        self.candidate_count = 0       # so frame lien tuc cua candidate
        self.STABILITY_FRAMES = 4      # can 4 frame lien tuc moi chuyen class (non-critical)

        # Hysteresis: can nhieu frame hon de ROI KHOI trang thai nguy hiem,
        # it frame hon de VAO trang thai nguy hiem
        self.STABILITY_ENTER = {       
            0: 2,  # DangerousDriving  
            1: 6,  # Distracted        
            2: 4,  # Drinking         
            3: 3,  # Safe              
            4: 2,  # SleepyDriving     
            5: 3,  # Yawn              
        }
        self.STABILITY_EXIT = {        
            0: 5,  # DangerousDriving  
            1: 3,  # Distracted        
            2: 3,  # Drinking
            4: 5,  # SleepyDriving     
            5: 4,  # Yawn
        }

        # Alert system
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        
        print("🚀 Initializing Improved Driver Drowsiness Detection System...")
        self._load_model()
        self._load_detectors()
        
    def _load_model(self):
        """Load the trained model with correct architecture matching weights"""
        # Auto-detect variant from filename
        if "B1" in os.path.basename(self.model_path).upper():
            variant = "B1"
            efficientnet_cls = tf.keras.applications.EfficientNetB1
        else:
            variant = "B0"
            efficientnet_cls = tf.keras.applications.EfficientNetB0

        # Training used 224x224 for ALL variants (see notebook cell-18)
        input_size = 224
        print(f"🔄 Loading EfficientNet-{variant} model ({input_size}x{input_size})...")

        base_model = efficientnet_cls(
            weights=None,
            include_top=False,
            input_shape=(input_size, input_size, 3),
        )
        # DO NOT pre-build with tf.zeros() — it creates variable names that
        # differ from the training checkpoint, causing skip_mismatch to silently
        # drop the Normalization layer weights (mean/variance).
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
        # DO NOT pre-build the Sequential model either.

        if os.path.exists(self.model_path):
            # Load WITHOUT skip_mismatch so all weights (including
            # Normalization stats) are loaded correctly.
            self.model.load_weights(self.model_path)
            print(f"✅ Model weights loaded from {self.model_path}")

            # Validate: check Normalization layer inside the base model
            base_layer = self.model.layers[0]  # EfficientNet base
            first_weights = base_layer.get_weights()
            if first_weights:
                w = first_weights[0]
                if np.max(np.abs(w)) == 0:
                    print("⚠️  WARNING: Base model Normalization weights are all zero!")
                    print("    Model may not work correctly. Check weight file compatibility.")
                else:
                    print(f"✅ Normalization weights OK (mean abs={np.mean(np.abs(w)):.4f})")
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

    def calculate_face_direction(self, landmarks, frame_shape):
        """Tinh huong nhin cua khuon mat tu FaceMesh landmarks.

        Dung do doi xung trai-phai cua khuon mat de xac dinh
        nguoi dang nhin thang hay quay sang ben.

        Nose tip (1) nam giua left cheek (234) va right cheek (454).
        Neu nhin thang: nose o giua -> ratio ~ 0.5
        Neu quay trai: nose gan left cheek -> ratio < 0.5
        Neu quay phai: nose gan right cheek -> ratio > 0.5

        Returns:
            face_ratio (float): 0.5 = thang, <0.4 hoac >0.6 = quay dau
            is_facing_forward (bool): True neu dang nhin thang
        """
        h, w = frame_shape[:2]

        nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
        left_cheek = np.array([landmarks[234].x * w, landmarks[234].y * h])
        right_cheek = np.array([landmarks[454].x * w, landmarks[454].y * h])

        face_width = np.linalg.norm(right_cheek - left_cheek) + 1e-6
        nose_to_left = np.linalg.norm(nose - left_cheek)

        # ratio: 0 = nose sat left, 1 = nose sat right, 0.5 = giua
        face_ratio = nose_to_left / face_width

        # Nhin thang: ratio trong khoang 0.35 - 0.65
        is_facing_forward = 0.35 < face_ratio < 0.65

        return face_ratio, is_facing_forward

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

        # Face direction
        face_ratio, is_facing_forward = self.calculate_face_direction(landmarks, frame.shape)

        return {
            'ear': avg_ear,
            'mar': mar,
            'is_drowsy': is_drowsy,
            'is_yawning': is_yawning,
            'is_facing_forward': is_facing_forward,
            'face_ratio': face_ratio,
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
        """Preprocess face image - exact match with training pipeline.

        Training order (notebook cell-11):
            1. decode JPEG → RGB (uint8)
            2. crop bounding box
            3. rgb_to_grayscale  →  concat ×3          ← grayscale FIRST
            4. tf.image.resize((224, 224))              ← resize SECOND
            5. float32 / 255.0

        We replicate this with TensorFlow ops so the numerical result
        is identical (same grayscale coefficients, same bilinear kernel).
        """
        # face_img is BGR uint8 from OpenCV
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Use TF ops to match training exactly
        img_tensor = tf.constant(face_rgb, dtype=tf.uint8)

        # 1. Grayscale BEFORE resize (matches training order)
        gray = tf.image.rgb_to_grayscale(img_tensor)          # (H, W, 1)
        gray_3ch = tf.concat([gray, gray, gray], axis=-1)     # (H, W, 3)

        # 2. Resize with TF bilinear (same as tf.image.resize in training)
        resized = tf.image.resize(gray_3ch, (224, 224))        # float32

        # 3. Normalize to [0, 1]
        normalized = resized / 255.0

        face_batch = tf.expand_dims(normalized, axis=0)        # (1, 224, 224, 3)
        return face_batch.numpy()
    
    def fuse_prediction(self, predicted_class, confidence, facial_metrics):
        """
        Ket hop ket qua model CNN voi chi so EAR/MAR/Face Direction.

        QUAN TRONG: Model bi bias nang sang Distracted khi dung webcam.
        Dung sensor (EAR + face direction) de override:
        - Mat mo + nhin thang + mieng binh thuong = SAFE bat ke model noi gi.
        - Chi tin model bao Distracted khi nguoi that su QUAY DAU di.
        """
        if facial_metrics is None:
            return predicted_class, confidence

        ear_drowsy = facial_metrics['is_drowsy']
        mar_yawning = facial_metrics['is_yawning']
        is_facing_forward = facial_metrics.get('is_facing_forward', True)

        # ======================================================
        # RULE 1 (HIGHEST PRIORITY): Mat mo + nhin thang = SAFE
        # Override model Distracted khi sensor chung minh dang tap trung
        # ======================================================
        if predicted_class == 1 and is_facing_forward and not ear_drowsy and not mar_yawning:
            return 3, confidence  # Ep ve SafeDriving

        # ======================================================
        # RULE 2: Boost khi sensor xac nhan model
        # ======================================================
        if predicted_class == 4 and ear_drowsy:
            confidence = min(confidence * 1.3, 1.0)
        if predicted_class == 5 and mar_yawning:
            confidence = min(confidence * 1.3, 1.0)

        # ======================================================
        # RULE 3: Downgrade khi sensor khong dong y (confidence thap)
        # ======================================================
        if predicted_class == 4 and not ear_drowsy:
            if confidence < 0.60:
                return 3, confidence
        if predicted_class == 5 and not mar_yawning:
            if confidence < 0.60:
                return 3, confidence

        # ======================================================
        # RULE 4: Override khi sensor phat hien nhung model khong nhan ra
        # ======================================================
        # EAR thap lien tuc -> buon ngu
        if self.drowsy_frames > self.DROWSY_FRAMES_THRESHOLD // 2 and predicted_class == 3:
            return 4, confidence

        # MAR cao -> ngap
        if mar_yawning and predicted_class == 3:
            return 5, confidence

        return predicted_class, confidence

    def smooth_predictions(self, prediction):
        """Smooth predictions using exponentially-weighted moving average.

        Recent frames get higher weight so the display reacts quickly
        while still filtering single-frame noise.
        alpha = 0.45 means current frame contributes 45 %, history 55 %.
        Gia tri nay giam nhay cam voi noise trong khi van phan ung kip.
        """
        self.prediction_history.append(prediction)

        if len(self.prediction_history) < 2:
            return prediction

        alpha = 0.45  # giam tu 0.6 -> 0.45 de loc noise tot hon
        recent = np.array(list(self.prediction_history))

        # Build exponential weights: oldest → newest
        n = len(recent)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        smoothed = np.average(recent, axis=0, weights=weights)
        return smoothed

    def stabilize_class(self, predicted_class, confidence):
        """Hysteresis-based stability filter voi confidence gate.

        - Khi CHUYEN VAO class moi: can STABILITY_ENTER[class] frame lien tuc
          VA confidence phai vuot nguong toi thieu.
        - Khi ROI KHOI class hien tai: can STABILITY_EXIT[class] frame lien tuc.
        - Distracted (1) can confidence >= 0.45 moi duoc xet, tranh false positive
          khi ngoi im nhin thang.
        """
        # Confidence gate: class nay can confidence toi thieu bao nhieu
        # moi duoc xet la "candidate" hop le
        MIN_CONFIDENCE = {
            0: 0.30,  # DangerousDriving  -> nguy hiem, nguong thap
            1: 0.45,  # Distracted        -> HAY BI FALSE POSITIVE, nguong cao
            2: 0.40,  # Drinking
            3: 0.25,  # Safe              -> de ve safe
            4: 0.30,  # SleepyDriving     -> nguy hiem, nguong thap
            5: 0.35,  # Yawn
        }

        min_conf = MIN_CONFIDENCE.get(predicted_class, 0.35)

        # Neu confidence khong du -> coi nhu giu nguyen class cu
        if confidence < min_conf:
            # Reset candidate neu dang dem cho class nay
            if predicted_class == self.candidate_class:
                self.candidate_class = self.stable_class
                self.candidate_count = 0
            return self.stable_class, self.stable_confidence

        if predicted_class == self.candidate_class:
            self.candidate_count += 1
        else:
            self.candidate_class = predicted_class
            self.candidate_count = 1

        # Giu nguyen class hien tai -> khong can lam gi
        if predicted_class == self.stable_class:
            return self.stable_class, self.stable_confidence

        # Dang muon chuyen sang class moi
        exit_frames = self.STABILITY_EXIT.get(self.stable_class, self.STABILITY_FRAMES)
        enter_frames = self.STABILITY_ENTER.get(predicted_class, self.STABILITY_FRAMES)
        required = max(enter_frames, exit_frames)

        if self.candidate_count >= required:
            self.stable_class = self.candidate_class
            self.stable_confidence = confidence

        return self.stable_class, self.stable_confidence
    
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

    def _play_alert_sound(self, predicted_class):
        """Play alert sound on background thread (non-blocking).

        Sound patterns:
        - Critical (class 0, 4): 3 rapid high beeps (urgent)
        - Warning  (class 1, 5): 2 medium beeps
        - Drowsy frames long:    continuous alarm tone
        """
        if not HAS_WINSOUND:
            return

        def _beep():
            try:
                is_critical = predicted_class in (0, 4)
                is_drowsy_long = self.drowsy_frames > self.DROWSY_FRAMES_THRESHOLD

                if is_critical or is_drowsy_long:
                    # Urgent: 3 rapid high-pitch beeps
                    for _ in range(3):
                        winsound.Beep(1200, 200)   # 1200 Hz, 200ms
                        time.sleep(0.05)
                else:
                    # Warning: 2 medium beeps
                    for _ in range(2):
                        winsound.Beep(800, 250)    # 800 Hz, 250ms
                        time.sleep(0.1)
            except Exception:
                pass  # Ignore sound errors silently

        # Run on daemon thread so it doesn't block video loop
        t = threading.Thread(target=_beep, daemon=True)
        t.start()

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
    
    # ================================================================
    #  DASHBOARD UI — Professional display for NCKH demo
    # ================================================================

    def _draw_semi_rect(self, frame, pt1, pt2, color, alpha=0.7):
        """Draw semi-transparent filled rectangle"""
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_corner_bbox(self, frame, x, y, fw, fh, color, thickness=2):
        """Draw sci-fi corner bracket bounding box"""
        corner_len = max(15, min(fw, fh) // 5)

        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x + fw, y), (x + fw - corner_len, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + fw, y), (x + fw, y + corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x, y + fh), (x + corner_len, y + fh), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + fh), (x, y + fh - corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, thickness, cv2.LINE_AA)

        # Thin connecting lines between corners (subtle)
        thin = max(1, thickness - 1)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, thin, cv2.LINE_AA)

    def _draw_header(self, frame):
        """Draw top header bar with system name, FPS, time"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent dark header (40px tall)
        self._draw_semi_rect(frame, (0, 0), (w, 42), (20, 20, 20), 0.80)

        # Accent line at bottom of header
        cv2.line(frame, (0, 42), (w, 42), (0, 200, 200), 1, cv2.LINE_AA)

        # Left: System name
        cv2.putText(frame, "AI DRIVER MONITORING SYSTEM",
                    (12, 28), font, 0.65, (0, 230, 230), 2, cv2.LINE_AA)

        # Right: FPS
        fps_text = ""
        if self.fps_counter:
            fps_val = np.mean(self.fps_counter)
            fps_text = f"FPS: {fps_val:.0f}"
            fps_color = (0, 255, 0) if fps_val >= 10 else (0, 165, 255)
            cv2.putText(frame, fps_text, (w - 260, 28), font, 0.55,
                        fps_color, 2, cv2.LINE_AA)

        # Right: inference time
        if self.inference_times:
            ms = np.mean(self.inference_times) * 1000
            cv2.putText(frame, f"{ms:.0f}ms", (w - 175, 28), font, 0.50,
                        (180, 180, 180), 1, cv2.LINE_AA)

        # Right: Clock
        clock_text = time.strftime("%H:%M:%S")
        cv2.putText(frame, clock_text, (w - 110, 28), font, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_metrics_panel(self, frame, facial_metrics):
        """Draw left-side metrics panel with semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        px, py = 8, 52          # top-left of panel
        pw, line_h = 195, 26    # panel width, line height

        if facial_metrics is None:
            # No face detected — show message
            panel_h = 40
            self._draw_semi_rect(frame, (px, py), (px + pw, py + panel_h),
                                 (20, 20, 20), 0.65)
            cv2.putText(frame, "Khong phat hien khuon mat",
                        (px + 8, py + 26), font, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
            return

        # Compute how many rows
        rows = 7  # EAR, MAR, Blinks, Yawns, Direction, Ratio, + header
        if not self.is_calibrated:
            rows += 1
        panel_h = 8 + rows * line_h + 8

        # Panel background
        self._draw_semi_rect(frame, (px, py), (px + pw, py + panel_h),
                             (20, 20, 20), 0.65)

        # Panel title
        cy = py + 20
        cv2.putText(frame, "CHI SO KHUON MAT", (px + 8, cy), font, 0.45,
                    (0, 200, 200), 1, cv2.LINE_AA)
        cy += 4
        cv2.line(frame, (px + 8, cy), (px + pw - 8, cy), (60, 60, 60), 1)
        cy += line_h - 4

        # Calibration warning
        if not self.is_calibrated:
            remaining = self.CALIBRATION_FRAMES - len(self.calibration_ear_buffer)
            cv2.putText(frame, f"Calibrating... {remaining}f",
                        (px + 8, cy), font, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
            cy += line_h

        # EAR
        ear_val = facial_metrics['ear']
        ear_color = (0, 0, 255) if self.drowsy_frames > 3 else (0, 255, 100)
        cv2.putText(frame, "EAR:", (px + 8, cy), font, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{ear_val:.3f}", (px + 55, cy), font, 0.50,
                    ear_color, 2, cv2.LINE_AA)
        # Mini bar for EAR (0-0.5 range)
        bar_x = px + 115
        bar_w = 70
        bar_fill = int(bar_w * min(ear_val / 0.40, 1.0))
        cv2.rectangle(frame, (bar_x, cy - 10), (bar_x + bar_w, cy - 2),
                      (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, cy - 10), (bar_x + bar_fill, cy - 2),
                      ear_color, -1)
        cy += line_h

        # MAR
        mar_val = facial_metrics['mar']
        mar_color = (0, 140, 255) if facial_metrics['is_yawning'] else (0, 255, 100)
        mar_label = " NGAP!" if facial_metrics['is_yawning'] else ""
        cv2.putText(frame, "MAR:", (px + 8, cy), font, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{mar_val:.3f}{mar_label}", (px + 55, cy), font, 0.50,
                    mar_color, 2, cv2.LINE_AA)
        cy += line_h

        # Blink count
        cv2.putText(frame, "Nhay mat:", (px + 8, cy), font, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.blink_counter}", (px + 100, cy), font, 0.50,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cy += line_h

        # Yawn count
        cv2.putText(frame, "So lan ngap:", (px + 8, cy), font, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.yawn_counter}", (px + 120, cy), font, 0.50,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cy += line_h

        # Face direction
        is_forward = facial_metrics.get('is_facing_forward', True)
        if is_forward:
            dir_text = "Thang"
            dir_icon = "<>"
            dir_color = (0, 255, 100)
        else:
            dir_text = "Quay"
            dir_icon = "><"
            dir_color = (0, 140, 255)
        cv2.putText(frame, "Huong:", (px + 8, cy), font, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{dir_text} {dir_icon}", (px + 70, cy), font, 0.50,
                    dir_color, 2, cv2.LINE_AA)
        cy += line_h

        # Face ratio
        face_ratio = facial_metrics.get('face_ratio', 0.5)
        cv2.putText(frame, "Ratio:", (px + 8, cy), font, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{face_ratio:.2f}", (px + 70, cy), font, 0.50,
                    (200, 200, 200), 1, cv2.LINE_AA)
        # Mini indicator: dot shows position on 0-1 scale
        ind_x = px + 120
        ind_w = 65
        cv2.line(frame, (ind_x, cy - 5), (ind_x + ind_w, cy - 5),
                 (60, 60, 60), 2, cv2.LINE_AA)
        dot_x = ind_x + int(ind_w * face_ratio)
        cv2.circle(frame, (dot_x, cy - 5), 4, dir_color, -1, cv2.LINE_AA)

    def _draw_prob_bars(self, frame, all_probs, predicted_class):
        """Draw compact probability bars on right side"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        bar_x = w - 215
        bar_y_start = 56
        bar_w = 90
        bar_h = 14
        gap = 5

        # Panel background
        self._draw_semi_rect(frame, (bar_x - 8, bar_y_start - 22),
                             (w - 5, bar_y_start + 6 * (bar_h + gap) + 4),
                             (20, 20, 20), 0.60)

        cv2.putText(frame, "XAC SUAT DU DOAN", (bar_x - 2, bar_y_start - 8),
                    font, 0.38, (0, 200, 200), 1, cv2.LINE_AA)

        short_names = {
            0: 'Nguy hiem',
            1: 'Mat t.trung',
            2: 'Uong nuoc',
            3: 'An toan',
            4: 'Buon ngu',
            5: 'Ngap'
        }

        for cls_id, prob in enumerate(all_probs):
            yp = bar_y_start + cls_id * (bar_h + gap)
            filled = int(bar_w * float(prob))

            # Background bar
            cv2.rectangle(frame, (bar_x, yp), (bar_x + bar_w, yp + bar_h),
                          (50, 50, 50), -1)

            # Filled bar
            fg_color = alert_config[cls_id]['color']
            if cls_id == predicted_class:
                # Brighten selected class
                fg_color = tuple(min(255, c + 40) for c in fg_color)
            cv2.rectangle(frame, (bar_x, yp), (bar_x + filled, yp + bar_h),
                          fg_color, -1)

            # Highlight border for predicted class
            if cls_id == predicted_class:
                cv2.rectangle(frame, (bar_x - 1, yp - 1),
                              (bar_x + bar_w + 1, yp + bar_h + 1),
                              (255, 255, 255), 1)

            # Label + percentage
            label = f"{short_names[cls_id]}: {prob:.0%}"
            label_color = (255, 255, 255) if cls_id == predicted_class else (180, 180, 180)
            cv2.putText(frame, label, (bar_x + bar_w + 6, yp + bar_h - 2),
                        font, 0.35, label_color, 1, cv2.LINE_AA)

    def _draw_status_bar(self, frame, predicted_class, confidence):
        """Draw bottom status bar with current state"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        bar_h = 42

        color = alert_config[predicted_class]['color']
        level = alert_config[predicted_class]['level']
        class_name = class_names[predicted_class]

        # Semi-transparent status bar
        self._draw_semi_rect(frame, (0, h - bar_h), (w, h), color, 0.55)

        # Top accent line
        cv2.line(frame, (0, h - bar_h), (w, h - bar_h), color, 2, cv2.LINE_AA)

        # Status indicator circle
        cv2.circle(frame, (22, h - bar_h // 2), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (22, h - bar_h // 2), 6, color, -1, cv2.LINE_AA)

        # Main status text
        status_text = f"[{level}]  {class_name}  -  {confidence:.0%}"
        cv2.putText(frame, status_text, (42, h - bar_h // 2 + 6),
                    font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Right side: detection method
        method = "MediaPipe" if self.use_mediapipe else "Haar"
        cv2.putText(frame, f"Engine: {method}", (w - 180, h - bar_h // 2 + 5),
                    font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_danger_alert(self, frame, predicted_class, frame_count):
        """Draw flashing red border + banner for dangerous states"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Determine danger level
        is_critical = predicted_class in (0, 4)  # Dangerous driving, Sleepy
        is_warning = predicted_class in (1, 5)   # Distracted, Yawn
        is_drowsy_long = self.drowsy_frames > self.DROWSY_FRAMES_THRESHOLD

        if not (is_critical or is_drowsy_long or is_warning):
            return

        # Flash toggle (on for 4 frames, off for 4 frames)
        flash_on = (frame_count % 8) < 4

        if is_critical or is_drowsy_long:
            border_color = (0, 0, 255)      # Red
            banner_color = (0, 0, 180)
            if is_drowsy_long:
                banner_text = "CANH BAO: DANG BUON NGU!"
            else:
                banner_text = f"CANH BAO: {class_names[predicted_class].upper()}!"
        else:
            border_color = (0, 140, 255)    # Orange
            banner_color = (0, 100, 200)
            banner_text = f"CHU Y: {class_names[predicted_class].upper()}"

        # Flashing border
        if flash_on:
            thickness = 6
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)
            cv2.rectangle(frame, (thickness, thickness),
                          (w - 1 - thickness, h - 1 - thickness),
                          (255, 255, 255), 1)

        # Center banner (always visible when in danger state)
        (tw, th), _ = cv2.getTextSize(banner_text, font, 0.9, 2)
        banner_w = tw + 60
        banner_h = th + 30
        bx = (w - banner_w) // 2
        by = h // 2 - banner_h - 20

        self._draw_semi_rect(frame, (bx, by), (bx + banner_w, by + banner_h),
                             banner_color, 0.75)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h),
                      border_color, 2, cv2.LINE_AA)

        # Banner text
        text_x = bx + 30
        text_y = by + th + 12
        if flash_on:
            cv2.putText(frame, banner_text, (text_x, text_y), font, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, banner_text, (text_x, text_y), font, 0.9,
                        (200, 200, 200), 2, cv2.LINE_AA)

    def draw_ui(self, frame, faces, predictions_data, facial_metrics=None,
                frame_count=0):
        """Draw professional dashboard UI for NCKH demo"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Header bar
        self._draw_header(frame)

        # 2. Metrics panel (left side)
        self._draw_metrics_panel(frame, facial_metrics)

        # 3. Process each detected face
        current_class = 3
        current_conf = 0.0
        for i, ((x, y, fw, fh), pred_data) in enumerate(zip(faces, predictions_data)):
            predicted_class, confidence, all_probs = pred_data
            current_class = predicted_class
            current_conf = confidence

            color = alert_config[predicted_class]['color']

            # Sci-fi corner bracket bbox
            thickness = 3 if predicted_class in (0, 4) else 2
            self._draw_corner_bbox(frame, x, y, fw, fh, color, thickness)

            # Face label above bbox
            class_name = class_names[predicted_class]
            label_text = f"{class_name} ({confidence:.0%})"
            (lw, lh), _ = cv2.getTextSize(label_text, font, 0.55, 2)

            label_y = max(48, y - 8)
            self._draw_semi_rect(frame, (x, label_y - lh - 6),
                                 (x + lw + 12, label_y + 2), color, 0.70)
            cv2.putText(frame, label_text, (x + 6, label_y - 4), font, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Probability bars (right side)
            self._draw_prob_bars(frame, all_probs, predicted_class)

        # 4. Status bar (bottom)
        self._draw_status_bar(frame, current_class, current_conf)

        # 5. Danger alert (flashing border + banner) — drawn LAST so it's on top
        self._draw_danger_alert(frame, current_class, frame_count)
    
    def run(self):
        """Main application loop"""
        print("📹 Starting webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

                    # Stability filter: class phai on dinh nhieu frame moi chuyen
                    predicted_class, confidence = self.stabilize_class(
                        predicted_class, confidence
                    )
                    
                    predictions_data.append((predicted_class, confidence, smoothed_predictions))
                    
                    # Check for alerts
                    if self.should_alert(predicted_class, facial_metrics):
                        print(f"🚨 ALERT: {class_names[predicted_class]} detected!")
                        if facial_metrics and facial_metrics['is_drowsy']:
                            print(f"   Eyes closed for {self.drowsy_frames} frames!")
                        self._play_alert_sound(predicted_class)
                
                # Draw UI
                self.draw_ui(frame, faces, predictions_data, facial_metrics,
                             frame_count)
                
                # Calculate FPS
                frame_time = time.time() - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_counter.append(fps)
                
                cv2.imshow('AI Driver Monitoring System - NCKH 2025', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset counters
                    self.blink_counter = 0
                    self.yawn_counter = 0
                    self.drowsy_frames = 0
                    self.stable_class = 3
                    self.stable_confidence = 0.0
                    self.candidate_class = 3
                    self.candidate_count = 0
                    self.prediction_history.clear()
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
