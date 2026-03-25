import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import threading
import sys
import os

# Import modules from main.py
try:
    import main as driver_detection
except ImportError:
    # If main.py is not found, we might need to adjust path
    sys.path.append(os.getcwd())
    import main as driver_detection

class ModernDrowsinessApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1400x800")
        
        # --- THEME CONFIGURATION ---
        self.colors = {
            'bg': '#1e1e1e',          # Dark Grey Background
            'panel_bg': '#252526',    # Slightly lighter panel
            'text': '#ffffff',        # White text
            'accent': '#007acc',      # Blue accent
            'safe': '#2ecc71',        # Green for Safe
            'warning': '#f39c12',     # Orange for Warning
            'danger': '#e74c3c',      # Red for Danger
            'info': '#3498db',        # Blue for Info
            'border': '#3e3e42'
        }
        
        self.window.configure(bg=self.colors['bg'])
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure widget styles
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('Panel.TFrame', background=self.colors['panel_bg'])
        
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 18, 'bold'), foreground=self.colors['accent'])
        self.style.configure('Status.TLabel', font=('Segoe UI', 14))
        self.style.configure('Value.TLabel', font=('Segoe UI', 16, 'bold'))
        
        self.style.configure('TButton', font=('Segoe UI', 11))
        self.style.map('TButton', background=[('active', self.colors['accent'])])

        # --- DETECTOR SETUP ---
        print("Initializing Detector...")
        self.detector = driver_detection.ImprovedDriverDetector(
            use_mediapipe=True,
            use_facemesh=True
        )
        
        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Variables
        self.running = True
        self.delay = 15
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Layout
        self._setup_ui()
        
        # Start Loop
        self.update()

    def _setup_ui(self):
        # 1. Header
        header_frame = ttk.Frame(self.window)
        header_frame.pack(fill=tk.X, padx=20, pady=15)
        
        title = ttk.Label(header_frame, text="DRIVER SAFETY MONITOR SYSTEM", style='Header.TLabel')
        title.pack(side=tk.LEFT)
        
        self.fps_label = ttk.Label(header_frame, text="FPS: 0", font=('Consolas', 12))
        self.fps_label.pack(side=tk.RIGHT)

        # 2. Main Layout (Split Left/Right)
        main_content = ttk.Frame(self.window)
        main_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # LEFT SIDE: Video Feed
        video_container = ttk.Frame(main_content, style='Panel.TFrame', padding=2)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for video
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(video_container, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # RIGHT SIDE: Data Dashboard
        dashboard = ttk.Frame(main_content, width=400, style='Panel.TFrame', padding=10)
        dashboard.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        dashboard.pack_propagate(False) # Fixed width

        # --- STATUS INDICATOR (Big Box) ---
        self.status_frame = tk.Frame(dashboard, bg=self.colors['safe'], height=120)
        self.status_frame.pack(fill=tk.X, pady=(0, 20))
        self.status_frame.pack_propagate(False)
        
        self.lbl_status_main = tk.Label(self.status_frame, text="SAFE", 
                                      font=('Segoe UI', 32, 'bold'), 
                                      bg=self.colors['safe'], fg='white')
        self.lbl_status_main.pack(expand=True, fill=tk.BOTH)
        
        self.lbl_status_sub = tk.Label(self.status_frame, text="Driver is attentive", 
                                     font=('Segoe UI', 11), 
                                     bg=self.colors['safe'], fg='#f0f0f0')
        self.lbl_status_sub.pack(side=tk.BOTTOM, pady=8)

        # --- METRICS SECTION ---
        ttk.Label(dashboard, text="REAL-TIME METRICS", font=('Segoe UI', 10, 'bold'), foreground='#888').pack(fill=tk.X, pady=5)
        
        metrics_grid = ttk.Frame(dashboard, style='Panel.TFrame')
        metrics_grid.pack(fill=tk.X, pady=5)
        
        # Row 1: EAR / MAR
        self._create_metric_card(metrics_grid, "EAR (Eyes)", "val_ear", 0, 0)
        self._create_metric_card(metrics_grid, "MAR (Mouth)", "val_mar", 0, 1)
        
        # Row 2: Blinks / Yawns
        self._create_metric_card(metrics_grid, "Blinks", "val_blinks", 1, 0)
        self._create_metric_card(metrics_grid, "Yawns", "val_yawns", 1, 1)

        ttk.Separator(dashboard, orient='horizontal').pack(fill='x', pady=20)

        # --- CALIBRATION STATUS ---
        calib_frame = ttk.Frame(dashboard, style='Panel.TFrame')
        calib_frame.pack(fill=tk.X)
        ttk.Label(calib_frame, text="System Status:", background=self.colors['panel_bg']).pack(anchor=tk.W)
        self.lbl_calib = ttk.Label(calib_frame, text="Initializing...", font=('Segoe UI', 10, 'italic'), foreground='orange', background=self.colors['panel_bg'])
        self.lbl_calib.pack(anchor=tk.W)

        # --- CONTROLS ---
        btn_frame = ttk.Frame(dashboard, style='Panel.TFrame')
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="RESET COUNTERS", command=self.reset_counters).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="EXIT SYSTEM", command=self.on_closing).pack(fill=tk.X, pady=5)

    def _create_metric_card(self, parent, title, var_name, row, col):
        card = tk.Frame(parent, bg=self.colors['bg'], padx=10, pady=10)
        card.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        parent.grid_columnconfigure(col, weight=1)
        
        lbl_title = tk.Label(card, text=title, font=('Segoe UI', 9), bg=self.colors['bg'], fg='#aaa')
        lbl_title.pack(anchor=tk.W)
        
        lbl_val = tk.Label(card, text="--", font=('Segoe UI', 18, 'bold'), bg=self.colors['bg'], fg='white')
        lbl_val.pack(anchor=tk.W)
        
        setattr(self, var_name, lbl_val)

    def _on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    def reset_counters(self):
        self.detector.blink_counter = 0
        self.detector.yawn_counter = 0
        self.detector.drowsy_frames = 0
        self.val_blinks.config(text="0")
        self.val_yawns.config(text="0")

    def update_status_display(self, predicted_class, facial_metrics=None):
        # Determine Color and Text based on Class
        # 0: Dangerous, 1: Distracted, 2: Drinking, 3: Safe, 4: Sleepy, 5: Yawn
        
        # Override logic for Drowsy check
        is_drowsy = False
        if facial_metrics and facial_metrics['is_drowsy'] and self.detector.drowsy_frames > self.detector.DROWSY_FRAMES_THRESHOLD:
            is_drowsy = True

        state_config = {
            'bg': self.colors['safe'],
            'text': 'SAFE',
            'sub': 'Driver is attentive'
        }

        if is_drowsy or predicted_class == 4:
            state_config = {'bg': self.colors['danger'], 'text': 'DROWSY!', 'sub': 'WAKE UP NOW!'}
        elif predicted_class == 5: # Yawn
            state_config = {'bg': self.colors['warning'], 'text': 'YAWNING', 'sub': 'Fatigue detected'}
        elif predicted_class == 1: # Distracted
            state_config = {'bg': self.colors['warning'], 'text': 'DISTRACTED', 'sub': 'Focus on road'}
        elif predicted_class == 0: # Dangerous
            state_config = {'bg': self.colors['danger'], 'text': 'DANGER', 'sub': 'Erratic behavior'}
        elif predicted_class == 2: # Drinking
            state_config = {'bg': self.colors['info'], 'text': 'DRINKING', 'sub': 'Stay focused'}
            
        # Update UI
        self.status_frame.configure(bg=state_config['bg'])
        self.lbl_status_main.configure(text=state_config['text'], bg=state_config['bg'])
        self.lbl_status_sub.configure(text=state_config['sub'], bg=state_config['bg'])

    def update(self):
        if not self.running:
            return
            
        loop_start = time.time()
        
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                # 1. FACE DETECTION
                faces = []
                if self.detector.use_mediapipe:
                    faces = self.detector.detect_faces_mediapipe(frame)
                else:
                    faces = self.detector.detect_faces_haar(frame)
                
                facial_metrics = None
                prediction_result = 3 # Default Safe
                
                if len(faces) > 0:
                    # 2. LANDMARKS (Only if faces found)
                    if self.detector.use_facemesh:
                        lm = self.detector.detect_face_landmarks(frame)
                        if lm:
                            facial_metrics = self.detector.extract_facial_metrics(frame, lm)
                            # Draw clean landmarks (Blue for eyes, Red/Orange for mouth)
                            self.detector.draw_facial_features(frame, facial_metrics)

                    # 3. PREDICTION (For the largest face)
                    # Use the first face found for simplicity in UI
                    x, y, w, h = faces[0]
                    # Draw subtle box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        # Preprocess
                        face_input = self.detector.preprocess_face(face_roi)
                        # Predict
                        preds = self.detector.model.predict(face_input, verbose=0)
                        # Smooth
                        smoothed = self.detector.smooth_predictions(preds[0])
                        # Get Class
                        p_class = np.argmax(smoothed)
                        conf = float(smoothed[p_class])
                        # Fuse
                        prediction_result, _ = self.detector.fuse_prediction(p_class, conf, facial_metrics)
                        
                    except Exception as e:
                        print(f"Pred Error: {e}")
                
                # 4. UPDATE UI COMPONENTS
                self.update_status_display(prediction_result, facial_metrics)

                if self.detector.trigger_alert(prediction_result, facial_metrics):
                    print(f"🚨 ALERT: {driver_detection.class_names[prediction_result]} detected!")
                
                # Update Metrics
                if facial_metrics:
                    self.val_ear.config(text=f"{facial_metrics['ear']:.2f}", fg='white')
                    self.val_mar.config(text=f"{facial_metrics['mar']:.2f}", fg='white')
                else:
                    self.val_ear.config(text="--", fg='#555')
                    self.val_mar.config(text="--", fg='#555')
                    
                self.val_blinks.config(text=str(self.detector.blink_counter))
                self.val_yawns.config(text=str(self.detector.yawn_counter))
                
                # Update Calibration Text
                if self.detector.is_calibrated:
                    self.lbl_calib.config(text="● Calibrated & Monitoring", foreground=self.colors['safe'])
                else:
                    rem = self.detector.CALIBRATION_FRAMES - len(self.detector.calibration_ear_buffer)
                    self.lbl_calib.config(text=f"● Calibrating... Keep eyes open ({rem})", foreground='orange')

                # 5. DRAW FRAME TO CANVAS
                # Resize keeping aspect ratio
                img_h, img_w = frame.shape[:2]
                
                # Calculate scale
                scale_w = self.canvas_width / img_w
                scale_h = self.canvas_height / img_h
                scale = min(scale_w, scale_h)
                
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                
                resized_frame = cv2.resize(frame, (new_w, new_h))
                img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_rgb))
                
                self.canvas.delete("all")
                # Center
                x_center = (self.canvas_width - new_w) // 2
                y_center = (self.canvas_height - new_h) // 2
                self.canvas.create_image(x_center, y_center, image=self.photo, anchor=tk.NW)

                # FPS Calculation
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        except Exception as e:
            print(f"Loop Error: {e}")

        if self.running:
            self.window.after(self.delay, self.update)

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()
        print("Application Closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernDrowsinessApp(root, "Modern Driver Safety System")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
