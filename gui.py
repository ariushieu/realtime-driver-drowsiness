"""
Driver Drowsiness Detection — Modern Dashboard GUI
Design: Video-dominant layout, compact right panel, dark theme.
"""

import os
import sys
import time
import io
import json
import threading
import unicodedata
from collections import deque
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import cv2
import numpy as np
import PIL.Image
import PIL.ImageTk
import tkinter as tk

try:
    import main as driver_detection
except ImportError:
    sys.path.append(os.getcwd())
    import main as driver_detection


# ─── Color Palette ───────────────────────────────────────────────
COLORS = {
    "bg":         "#0B1530",
    "panel":      "#1A2748",
    "panel_soft": "#202F54",
    "text_main":  "#EEF5FF",
    "text_sub":   "#95A8CB",
    "accent":     "#35D8F6",
    "safe":       "#37C978",
    "warning":    "#F1B53F",
    "danger":     "#E6576F",
    "drinking":   "#3A9CFF",
    "border":     "#2D416D",
    "bar_bg":     "#2A3B64",
}

CLASS_NAMES_VI = {
    0: "Lái xe nguy hiểm",
    1: "Mất tập trung",
    2: "Đang uống nước",
    3: "Lái xe an toàn",
    4: "Buồn ngủ",
    5: "Ngáp",
}

CLASS_COLORS = {
    0: COLORS["danger"],
    1: COLORS["warning"],
    2: COLORS["drinking"],
    3: COLORS["safe"],
    4: COLORS["danger"],
    5: COLORS["warning"],
}


class ModernDrowsinessApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1600x900")
        self.window.minsize(1280, 760)
        try:
            self.window.state("zoomed")
        except Exception:
            pass
        self.window.configure(bg=COLORS["bg"])

        # ── Detector ─────────────────────────────────────────────
        print("Đang khởi tạo bộ phát hiện...")
        self.detector = driver_detection.ImprovedDriverDetector(
            use_mediapipe=True,
            use_facemesh=True,
        )

        # ── Camera ───────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ── Timing / FPS ─────────────────────────────────────────
        self.running = True
        self.delay = 15
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_inference_ms = 0.0

        # ── Prediction state ─────────────────────────────────────
        self.current_prediction = 3
        self.current_confidence = 0.0
        self.current_probs = np.zeros(6, dtype=np.float32)

        # ── Sparkline data ───────────────────────────────────────
        self.ear_history = deque(maxlen=40)
        self.mar_history = deque(maxlen=40)

        # ── Alert system ─────────────────────────────────────────
        self.alert_events = deque(maxlen=8)
        self.last_event_key = None
        self.last_event_time = 0.0
        self.event_cooldowns = {
            "danger": 2.0, "distracted": 3.0, "drowsy": 2.0,
            "yawn": 4.0, "no_face": 6.0, "face_back": 4.0,
            "resolved": 8.0, "system": 12.0,
            "location": 20.0, "location_error": 30.0,
        }
        self.last_event_emit = {}
        self.no_face_start_time = None
        self.no_face_reported = False
        self.had_recent_warning = False
        self.last_alert_panel_refresh = 0.0

        # ── Widget refs (populated by builders) ──────────────────
        self.prob_bars = []
        self.alert_widgets = []
        self.canvas_width = 900
        self.canvas_height = 520

        # ── Map state ────────────────────────────────────────────
        self.map_center_lat = 21.0285
        self.map_center_lon = 105.8542
        self.map_zoom = 13
        self.map_bg_cache = None
        self.map_bg_size = (0, 0)
        self.map_photo = None
        self.map_pulse_phase = 0
        self.map_last_refresh = 0.0

        # ── Location service ─────────────────────────────────────
        self.location_enabled = True
        self.location_refresh_interval = 180.0
        self.location_last_update = 0.0
        self.location_fetch_in_progress = False
        self.location_name = "Đang xác định vị trí..."

        # ── Build UI & start loops ───────────────────────────────
        self._setup_ui()
        self._seed_alerts()
        self._request_location_update(force=True)
        self._schedule_map_animation()
        self.update()

    # ═══════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════

    def _setup_ui(self):
        self._build_header()
        self._build_body()

    # ── Header ────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self.window, bg=COLORS["bg"], padx=18, pady=8)
        header.pack(fill=tk.X)

        # Left: title
        title_fr = tk.Frame(header, bg=COLORS["bg"])
        title_fr.pack(side=tk.LEFT)
        tk.Label(
            title_fr, text="AI GIÁM SÁT LÁI XE",
            font=("Segoe UI", 24, "bold"),
            fg=COLORS["text_main"], bg=COLORS["bg"],
        ).pack(side=tk.LEFT)
        tk.Label(
            title_fr, text="  ●  Trực tiếp",
            font=("Segoe UI", 12),
            fg=COLORS["safe"], bg=COLORS["bg"],
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Right: clock + fps
        info_fr = tk.Frame(header, bg=COLORS["bg"])
        info_fr.pack(side=tk.RIGHT)
        self.lbl_clock = tk.Label(
            info_fr, text="--:--",
            font=("Segoe UI", 20, "bold"),
            fg=COLORS["text_main"], bg=COLORS["bg"],
        )
        self.lbl_clock.pack(anchor=tk.E)
        self.lbl_fps = tk.Label(
            info_fr, text="FPS: -- | --ms",
            font=("Consolas", 11),
            fg=COLORS["safe"], bg=COLORS["bg"],
        )
        self.lbl_fps.pack(anchor=tk.E)

    # ── Body (left col + right col) ──────────────────────────────

    def _build_body(self):
        body = tk.Frame(self.window, bg=COLORS["bg"], padx=14, pady=(0,))
        body.pack(fill=tk.BOTH, expand=True)

        # Right column first (fixed width) so left gets remaining space
        self.right_col = tk.Frame(body, bg=COLORS["bg"], width=350)
        self.right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.right_col.pack_propagate(False)

        # Left column (expand to fill)
        left_col = tk.Frame(body, bg=COLORS["bg"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Left: prob bars at bottom FIRST, then video fills rest
        self._build_probability_bars(left_col)
        self._build_video_card(left_col)

        # Right: status → metrics → map → alerts
        self._build_status_box(self.right_col)
        self._build_metrics_panel(self.right_col)
        self._build_map(self.right_col)
        self._build_alert_log(self.right_col)

    # ── Video Card ───────────────────────────────────────────────

    def _build_video_card(self, parent):
        card = tk.Frame(
            parent, bg=COLORS["panel"],
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.canvas = tk.Canvas(card, bg="#08101F", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Probability Bars (2×3 grid) ──────────────────────────────

    def _build_probability_bars(self, parent):
        frame = tk.Frame(
            parent, bg=COLORS["panel"],
            highlightbackground=COLORS["border"], highlightthickness=1,
            padx=12, pady=8,
        )
        frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))

        for c in range(3):
            frame.columnconfigure(c, weight=1)

        self.prob_bars = []
        for cls_id in range(6):
            row_idx = cls_id // 3
            col_idx = cls_id % 3

            cell = tk.Frame(frame, bg=COLORS["panel"], padx=4, pady=2)
            cell.grid(row=row_idx, column=col_idx, sticky="ew")

            label = tk.Label(
                cell, text=f"{CLASS_NAMES_VI[cls_id]}: 0%",
                font=("Segoe UI", 9), anchor="w",
                fg=COLORS["text_sub"], bg=COLORS["panel"],
            )
            label.pack(fill=tk.X)

            bar = tk.Canvas(cell, height=8, bg=COLORS["bar_bg"], highlightthickness=0)
            bar.pack(fill=tk.X)
            rect = bar.create_rectangle(0, 0, 0, 8, fill=CLASS_COLORS[cls_id], width=0)

            self.prob_bars.append({"canvas": bar, "rect": rect, "label": label, "cls": cls_id})

    # ── Status Box ───────────────────────────────────────────────

    def _build_status_box(self, parent):
        self.status_box = tk.Frame(
            parent, bg=COLORS["safe"], padx=14, pady=12,
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        self.status_box.pack(fill=tk.X, pady=(0, 8))

        self.lbl_status_main = tk.Label(
            self.status_box, text="AN TOÀN",
            font=("Segoe UI", 28, "bold"),
            fg="#EFFCF4", bg=COLORS["safe"],
        )
        self.lbl_status_main.pack(anchor=tk.W)

        self.lbl_status_sub = tk.Label(
            self.status_box, text="Tài xế đang tập trung",
            font=("Segoe UI", 12),
            fg="#E8FFF0", bg=COLORS["safe"],
        )
        self.lbl_status_sub.pack(anchor=tk.W)

    # ── Metrics Panel ────────────────────────────────────────────

    def _build_metrics_panel(self, parent):
        panel = tk.Frame(
            parent, bg=COLORS["panel"], padx=14, pady=10,
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        panel.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            panel, text="Chẩn đoán",
            font=("Segoe UI", 13, "bold"),
            fg=COLORS["text_main"], bg=COLORS["panel"],
        ).pack(anchor=tk.W, pady=(0, 6))

        # EAR row + sparkline
        ear_row = tk.Frame(panel, bg=COLORS["panel"])
        ear_row.pack(fill=tk.X, pady=1)
        self.lbl_ear = tk.Label(
            ear_row, text="EAR: --",
            font=("Consolas", 11), fg=COLORS["accent"], bg=COLORS["panel"],
        )
        self.lbl_ear.pack(side=tk.LEFT)
        self.chart_ear = tk.Canvas(ear_row, height=22, bg=COLORS["panel"], highlightthickness=0)
        self.chart_ear.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        # MAR row + sparkline
        mar_row = tk.Frame(panel, bg=COLORS["panel"])
        mar_row.pack(fill=tk.X, pady=1)
        self.lbl_mar = tk.Label(
            mar_row, text="MAR: --",
            font=("Consolas", 11), fg="#B9C5E6", bg=COLORS["panel"],
        )
        self.lbl_mar.pack(side=tk.LEFT)
        self.chart_mar = tk.Canvas(mar_row, height=22, bg=COLORS["panel"], highlightthickness=0)
        self.chart_mar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        # Separator
        tk.Frame(panel, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=6)

        # Blinks / Yawns
        stats_row = tk.Frame(panel, bg=COLORS["panel"])
        stats_row.pack(fill=tk.X, pady=1)
        tk.Label(stats_row, text="Chớp mắt:", font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"]).pack(side=tk.LEFT)
        self.lbl_blinks = tk.Label(stats_row, text="0", font=("Segoe UI", 10, "bold"), fg=COLORS["text_main"], bg=COLORS["panel"])
        self.lbl_blinks.pack(side=tk.LEFT, padx=(4, 16))
        tk.Label(stats_row, text="Ngáp:", font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"]).pack(side=tk.LEFT)
        self.lbl_yawns = tk.Label(stats_row, text="0", font=("Segoe UI", 10, "bold"), fg=COLORS["text_main"], bg=COLORS["panel"])
        self.lbl_yawns.pack(side=tk.LEFT, padx=(4, 0))

        # Face direction
        face_row = tk.Frame(panel, bg=COLORS["panel"])
        face_row.pack(fill=tk.X, pady=1)
        tk.Label(face_row, text="Hướng mặt:", font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"]).pack(side=tk.LEFT)
        self.lbl_face_dir = tk.Label(face_row, text="--", font=("Segoe UI", 10, "bold"), fg=COLORS["text_sub"], bg=COLORS["panel"])
        self.lbl_face_dir.pack(side=tk.LEFT, padx=(4, 16))
        tk.Label(face_row, text="Tỷ lệ:", font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"]).pack(side=tk.LEFT)
        self.lbl_face_ratio = tk.Label(face_row, text="--", font=("Consolas", 10), fg=COLORS["text_sub"], bg=COLORS["panel"])
        self.lbl_face_ratio.pack(side=tk.LEFT, padx=(4, 0))

        # Separator
        tk.Frame(panel, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=6)

        # Calibration + location
        self.lbl_calib = tk.Label(
            panel, text="Hiệu chuẩn: Đang khởi tạo...",
            font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"],
        )
        self.lbl_calib.pack(anchor=tk.W)
        self.lbl_location_status = tk.Label(
            panel, text="Vị trí: Đang xác định...",
            font=("Segoe UI", 10), fg=COLORS["text_sub"], bg=COLORS["panel"],
        )
        self.lbl_location_status.pack(anchor=tk.W)

    # ── Mini Map ─────────────────────────────────────────────────

    def _build_map(self, parent):
        map_card = tk.Frame(
            parent, bg=COLORS["panel"], height=160,
            highlightbackground=COLORS["border"], highlightthickness=1,
            padx=8, pady=6,
        )
        map_card.pack(fill=tk.X, pady=(0, 8))
        map_card.pack_propagate(False)

        # Header row
        head = tk.Frame(map_card, bg=COLORS["panel"])
        head.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            head, text="Bản đồ",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_main"], bg=COLORS["panel"],
        ).pack(side=tk.LEFT)
        tk.Button(
            head, text="↻",
            command=lambda: self._request_location_update(force=True),
            font=("Segoe UI", 10, "bold"),
            fg=COLORS["text_main"], bg=COLORS["panel_soft"],
            activebackground=COLORS["accent"], activeforeground=COLORS["bg"],
            relief=tk.FLAT, padx=6, cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Subtitle
        sub = tk.Frame(map_card, bg=COLORS["panel"])
        sub.pack(fill=tk.X)
        self.lbl_map_place = tk.Label(
            sub, text=self.location_name,
            font=("Segoe UI", 9), fg=COLORS["text_sub"], bg=COLORS["panel"], anchor="w",
        )
        self.lbl_map_place.pack(side=tk.LEFT)
        self.lbl_map_coord = tk.Label(
            sub, text=f"{self.map_center_lat:.4f}, {self.map_center_lon:.4f}",
            font=("Consolas", 9), fg=COLORS["text_sub"], bg=COLORS["panel"],
        )
        self.lbl_map_coord.pack(side=tk.RIGHT)

        # Canvas
        self.map_canvas = tk.Canvas(map_card, bg="#1A2748", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True)
        self.map_canvas.bind("<Configure>", self._draw_map)

    # ── Alert Log ────────────────────────────────────────────────

    def _build_alert_log(self, parent):
        frame = tk.Frame(
            parent, bg=COLORS["panel"], padx=10, pady=8,
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            frame, text="Cảnh báo gần đây",
            font=("Segoe UI", 12, "bold"),
            fg=COLORS["text_main"], bg=COLORS["panel"],
        ).pack(anchor=tk.W, pady=(0, 6))

        self.alert_list = tk.Frame(frame, bg=COLORS["panel"])
        self.alert_list.pack(fill=tk.BOTH, expand=True)

    # ═══════════════════════════════════════════════════════════════
    #  MAP SYSTEM (ported from original)
    # ═══════════════════════════════════════════════════════════════

    def _build_map_url(self, width, height):
        size_w = max(200, min(width, 650))
        size_h = max(140, min(height, 450))
        query = urlencode({
            "ll": f"{self.map_center_lon:.6f},{self.map_center_lat:.6f}",
            "z": str(self.map_zoom),
            "size": f"{size_w},{size_h}",
            "l": "map",
            "lang": "en_US",
        })
        return f"https://static-maps.yandex.ru/1.x/?{query}"

    def _request_location_update(self, force=False):
        if not self.location_enabled or self.location_fetch_in_progress:
            return
        now = time.time()
        if not force and (now - self.location_last_update) < self.location_refresh_interval:
            return
        self.location_fetch_in_progress = True
        self.location_last_update = now
        threading.Thread(target=self._location_lookup_worker, daemon=True).start()

    def _location_lookup_worker(self):
        endpoints = ["https://ipapi.co/json/", "https://ipwho.is/"]
        last_error = ""
        for endpoint in endpoints:
            try:
                with urlopen(endpoint, timeout=3.5) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="ignore"))
                lat = payload.get("latitude", payload.get("lat"))
                lon = payload.get("longitude", payload.get("lon"))
                if lat is None or lon is None:
                    raise ValueError("Không nhận được tọa độ")
                city = payload.get("city") or payload.get("district") or "Không rõ"
                region = payload.get("region") or payload.get("region_name") or ""
                country = payload.get("country_name") or payload.get("country") or ""
                loc = {"lat": float(lat), "lon": float(lon), "city": city, "region": region, "country": country}
                self.window.after(0, lambda p=loc: self._apply_location_update(p))
                return
            except Exception as exc:
                last_error = str(exc)
        self.window.after(0, lambda: self._handle_location_error(last_error))

    def _apply_location_update(self, payload):
        self.location_fetch_in_progress = False
        self.map_center_lat = payload["lat"]
        self.map_center_lon = payload["lon"]
        self.map_bg_cache = None
        self.map_bg_size = (0, 0)
        area = ", ".join(p for p in [payload.get("city", ""), payload.get("region", ""), payload.get("country", "")] if p)
        self.location_name = area or "Không rõ khu vực"
        self.lbl_map_place.config(text=self.location_name, fg=COLORS["safe"])
        self.lbl_map_coord.config(text=f"{self.map_center_lat:.5f}, {self.map_center_lon:.5f}")
        self.lbl_location_status.config(text=f"Vị trí: {self.location_name}", fg=COLORS["safe"])
        self._push_alert(f"Vị trí: {self.location_name}", "Hệ thống", "INFO", event_key="location")
        self._draw_map()

    def _handle_location_error(self, message):
        self.location_fetch_in_progress = False
        self.lbl_map_place.config(text="Ngoại tuyến", fg=COLORS["warning"])
        self.lbl_location_status.config(text="Vị trí: Ngoại tuyến", fg=COLORS["warning"])
        self._push_alert("Không kết nối dịch vụ định vị", "Hệ thống", "WARNING", event_key="location_error")

    def _try_fetch_map_background(self, width, height):
        if self.map_bg_cache is not None and self.map_bg_size == (width, height):
            return self.map_bg_cache
        try:
            with urlopen(self._build_map_url(width, height), timeout=2.5) as response:
                data = response.read()
            img = PIL.Image.open(io.BytesIO(data)).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
            self.map_bg_cache = img
            self.map_bg_size = (width, height)
            return img
        except Exception:
            return None

    def _draw_map_fallback_background(self, w, h):
        self.map_canvas.create_rectangle(0, 0, w, h, fill="#1A2748", outline="")
        for x in range(0, w + 1, max(28, w // 12)):
            self.map_canvas.create_line(x, 0, x, h, fill="#24385F", width=1)
        for y in range(0, h + 1, max(24, h // 10)):
            self.map_canvas.create_line(0, y, w, y, fill="#24385F", width=1)
        for i in range(6):
            x0 = int((i * 0.17 + 0.05) * w)
            y0 = int((0.12 + (i % 3) * 0.22) * h)
            x1 = min(w - 8, x0 + int(0.14 * w))
            y1 = min(h - 8, y0 + int(0.13 * h))
            self.map_canvas.create_rectangle(x0, y0, x1, y1, outline="#2D416D", fill="#182845", width=1)

    def _draw_map_pin(self, w, h):
        cx, cy = w // 2, h // 2
        pulse = 10 + (self.map_pulse_phase % 8) * 2
        self.map_canvas.create_oval(cx - pulse, cy - pulse, cx + pulse, cy + pulse, outline="#35D8F6", width=2)
        self.map_canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#35D8F6", outline="#D6FAFF", width=2)
        self.map_canvas.create_text(cx, min(h - 10, cy + 18), text="Vị trí xe", fill=COLORS["text_main"], font=("Segoe UI", 9, "bold"))

    def _draw_map(self, event=None):
        w = self.map_canvas.winfo_width()
        h = self.map_canvas.winfo_height()
        if w < 10 or h < 10:
            return
        self.map_canvas.delete("all")
        bg_image = self._try_fetch_map_background(w, h)
        if bg_image is not None:
            self.map_photo = PIL.ImageTk.PhotoImage(bg_image)
            self.map_canvas.create_image(0, 0, image=self.map_photo, anchor=tk.NW)
            self.map_canvas.create_rectangle(0, 0, w, h, fill="#0B1530", stipple="gray50", outline="")
        else:
            self._draw_map_fallback_background(w, h)
        self._draw_map_pin(w, h)

    def _schedule_map_animation(self):
        if not self.running:
            return
        self.map_pulse_phase = (self.map_pulse_phase + 1) % 24
        self._request_location_update(force=False)
        now = time.time()
        if now - self.map_last_refresh > 0.35:
            self._draw_map()
            self.map_last_refresh = now
        self.window.after(350, self._schedule_map_animation)

    # ═══════════════════════════════════════════════════════════════
    #  ALERT SYSTEM
    # ═══════════════════════════════════════════════════════════════

    def _seed_alerts(self):
        self._push_alert("Hệ thống đã khởi động", "Hệ thống", "INFO", event_key="system", force=True)
        self._push_alert("Camera 1 đã kết nối", "Hệ thống", "INFO", event_key="system", force=True)

    def _state_from_prediction(self, predicted_class, facial_metrics):
        is_drowsy = (
            facial_metrics
            and facial_metrics.get("is_drowsy")
            and self.detector.drowsy_frames > self.detector.DROWSY_FRAMES_THRESHOLD
        )
        state = {
            "key": "safe", "bg": COLORS["safe"],
            "title": "AN TOÀN", "subtitle": "Tài xế đang tập trung",
        }
        if is_drowsy or predicted_class == 4:
            state = {"key": "drowsy", "bg": COLORS["danger"], "title": "BUỒN NGỦ", "subtitle": "Hãy tỉnh táo ngay!"}
        elif predicted_class == 0:
            state = {"key": "danger", "bg": COLORS["danger"], "title": "NGUY HIỂM", "subtitle": "Hành vi lái xe bất thường"}
        elif predicted_class == 5:
            state = {"key": "yawn", "bg": COLORS["warning"], "title": "NGÁP", "subtitle": "Dấu hiệu mệt mỏi"}
        elif predicted_class == 1:
            state = {"key": "distracted", "bg": COLORS["warning"], "title": "MẤT TẬP TRUNG", "subtitle": "Hãy nhìn đường!"}
        elif predicted_class == 2:
            state = {"key": "drinking", "bg": COLORS["drinking"], "title": "ĐANG UỐNG", "subtitle": "Đã phát hiện hành động"}
        return state

    def _format_relative_time(self, created_at):
        delta = int(max(0, time.time() - created_at))
        if delta < 60:
            return f"{delta}s trước"
        if delta < 3600:
            return f"{delta // 60}m trước"
        return f"{delta // 3600}h trước"

    def _push_alert(self, title, actor, badge, event_key=None, force=False):
        now = time.time()
        if event_key and not force:
            last_emit = self.last_event_emit.get(event_key, 0.0)
            cooldown = self.event_cooldowns.get(event_key, 3.0)
            if now - last_emit < cooldown:
                return
            self.last_event_emit[event_key] = now

        badge_text = {"CRITICAL": "NGHIÊM TRỌNG", "WARNING": "CẢNH BÁO", "RESOLVED": "ĐÃ ỔN ĐỊNH", "INFO": "THÔNG TIN"}.get(badge, badge)
        badge_color = {"CRITICAL": COLORS["danger"], "WARNING": COLORS["warning"], "RESOLVED": COLORS["safe"], "INFO": COLORS["drinking"]}.get(badge, COLORS["text_sub"])

        self.alert_events.appendleft({
            "title": title, "actor": actor,
            "clock": datetime.now().strftime("%I:%M %p").lstrip("0"),
            "created_at": now,
            "badge_text": badge_text, "badge_color": badge_color,
            "event_key": event_key or "generic",
        })
        self._render_alert_cards()

    def _render_alert_cards(self):
        for w in self.alert_widgets:
            w.destroy()
        self.alert_widgets = []

        for event in list(self.alert_events)[:4]:
            card = tk.Frame(self.alert_list, bg=COLORS["panel_soft"], padx=8, pady=6)
            card.pack(fill=tk.X, pady=(0, 4))

            # Top: badge + time
            top = tk.Frame(card, bg=COLORS["panel_soft"])
            top.pack(fill=tk.X)
            tk.Label(
                top, text=event["badge_text"],
                font=("Segoe UI", 8, "bold"), fg="#F8FBFF",
                bg=event["badge_color"], padx=6, pady=1,
            ).pack(side=tk.LEFT)
            rel = self._format_relative_time(event.get("created_at", time.time()))
            tk.Label(
                top, text=f"{event.get('clock', '')} · {rel}",
                font=("Segoe UI", 8), fg=COLORS["text_sub"], bg=COLORS["panel_soft"],
            ).pack(side=tk.RIGHT)

            # Bottom: title
            tk.Label(
                card, text=event["title"],
                font=("Segoe UI", 9), fg=COLORS["text_main"],
                bg=COLORS["panel_soft"], anchor="w", wraplength=300,
            ).pack(fill=tk.X, pady=(2, 0))

            self.alert_widgets.append(card)

    def _emit_realtime_events(self, face_count, predicted_class, confidence, facial_metrics):
        now = time.time()

        # No face detection
        if face_count == 0:
            if self.no_face_start_time is None:
                self.no_face_start_time = now
            if (now - self.no_face_start_time) > 2.5 and not self.no_face_reported:
                self._push_alert("Không phát hiện khuôn mặt", "Camera 1", "WARNING", event_key="no_face")
                self.no_face_reported = True
        else:
            if self.no_face_reported:
                self._push_alert("Đã nhận lại khuôn mặt", "Camera 1", "RESOLVED", event_key="face_back")
            self.no_face_start_time = None
            self.no_face_reported = False

        is_drowsy = bool(
            facial_metrics and facial_metrics.get("is_drowsy")
            and self.detector.drowsy_frames > max(6, self.detector.DROWSY_FRAMES_THRESHOLD // 2)
        )
        is_yawning = bool(facial_metrics and facial_metrics.get("is_yawning"))

        if predicted_class == 0 and confidence >= 0.35:
            self._push_alert("Lái xe nguy hiểm", "AI", "CRITICAL", event_key="danger")
            self.had_recent_warning = True
        if predicted_class == 1 and confidence >= 0.35:
            self._push_alert("Mất tập trung", "AI", "WARNING", event_key="distracted")
            self.had_recent_warning = True
        if predicted_class == 4 or is_drowsy:
            self._push_alert(f"Buồn ngủ ({self.detector.drowsy_frames} frames)", "AI", "CRITICAL", event_key="drowsy")
            self.had_recent_warning = True
        if predicted_class == 5 or is_yawning:
            self._push_alert(f"Ngáp (tổng: {self.detector.yawn_counter})", "AI", "WARNING", event_key="yawn")
            self.had_recent_warning = True

        is_safe = face_count > 0 and predicted_class == 3 and not is_drowsy and not is_yawning
        if is_safe and self.had_recent_warning:
            self._push_alert("Trạng thái ổn định", "AI", "RESOLVED", event_key="resolved")
            self.had_recent_warning = False

    # ═══════════════════════════════════════════════════════════════
    #  DISPLAY UPDATES
    # ═══════════════════════════════════════════════════════════════

    def _on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    def _draw_sparkline(self, canvas, values, color):
        width = max(canvas.winfo_width(), 80)
        height = max(canvas.winfo_height(), 20)
        canvas.delete("all")
        if len(values) < 2:
            return
        arr = np.array(values, dtype=np.float32)
        min_v, max_v = float(arr.min()), float(arr.max())
        span = max(max_v - min_v, 1e-6)
        points = []
        for i, val in enumerate(arr):
            x = int(i * (width - 4) / (len(arr) - 1)) + 2
            y = int((1.0 - (val - min_v) / span) * (height - 6)) + 3
            points.extend([x, y])
        canvas.create_line(points, fill=color, width=2, smooth=True)

    def _to_cv_text(self, text):
        raw = str(text)
        safe = raw.replace("Đ", "D").replace("đ", "d")
        safe = unicodedata.normalize("NFKD", safe).encode("ascii", "ignore").decode("ascii")
        return safe if safe else raw

    def _update_probability_bars(self, probabilities):
        probs = probabilities if probabilities is not None else np.zeros(6, dtype=np.float32)
        for i, item in enumerate(self.prob_bars):
            p = float(probs[i]) if i < len(probs) else 0.0
            p = min(max(p, 0.0), 1.0)
            width = max(item["canvas"].winfo_width(), 10)
            item["canvas"].coords(item["rect"], 0, 0, int(width * p), 8)
            item["label"].config(text=f"{CLASS_NAMES_VI[i]}: {p:.0%}")

    def _draw_video_overlay(self, frame, faces, predicted_class, confidence, facial_metrics):
        h, w = frame.shape[:2]
        color_hex = CLASS_COLORS.get(predicted_class, COLORS["safe"])
        bgr = tuple(int(color_hex[i: i + 2], 16) for i in (5, 3, 1))

        if faces:
            x, y, fw, fh = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), bgr, 2)
            label = f"{CLASS_NAMES_VI.get(predicted_class, '?')} ({confidence:.0%})"
            cv2.putText(frame, self._to_cv_text(label), (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)

        cv2.putText(frame, self._to_cv_text(f"FPS: {self.fps:.1f}"), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (62, 255, 174), 2)

        method = "MediaPipe" if self.detector.use_mediapipe else "Haar"
        cv2.putText(frame, self._to_cv_text(method), (w - 160, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 245, 255), 2)

    def _update_status_display(self, predicted_class, facial_metrics, probabilities):
        state = self._state_from_prediction(predicted_class, facial_metrics)
        self.status_box.configure(bg=state["bg"])
        self.lbl_status_main.configure(text=state["title"], bg=state["bg"])
        self.lbl_status_sub.configure(text=state["subtitle"], bg=state["bg"])
        self._update_probability_bars(probabilities)

    def _update_metrics_display(self, facial_metrics):
        if facial_metrics:
            ear = float(facial_metrics["ear"])
            mar = float(facial_metrics["mar"])

            ear_color = COLORS["danger"] if self.detector.drowsy_frames > 3 else COLORS["accent"]
            self.lbl_ear.config(text=f"EAR: {ear:.3f}", fg=ear_color)

            mar_color = COLORS["warning"] if facial_metrics["is_yawning"] else "#B9C5E6"
            self.lbl_mar.config(text=f"MAR: {mar:.3f}", fg=mar_color)

            self.ear_history.append(ear)
            self.mar_history.append(mar)

            # Face direction
            is_forward = facial_metrics.get("is_facing_forward", True)
            if is_forward:
                self.lbl_face_dir.config(text="Thẳng ✓", fg=COLORS["safe"])
            else:
                self.lbl_face_dir.config(text="Quay ✗", fg=COLORS["warning"])

            face_ratio = facial_metrics.get("face_ratio", 0.5)
            self.lbl_face_ratio.config(text=f"{face_ratio:.2f}", fg=COLORS["text_main"])
        else:
            self.lbl_ear.config(text="EAR: --", fg=COLORS["text_sub"])
            self.lbl_mar.config(text="MAR: --", fg=COLORS["text_sub"])
            self.lbl_face_dir.config(text="--", fg=COLORS["text_sub"])
            self.lbl_face_ratio.config(text="--", fg=COLORS["text_sub"])

        self.lbl_blinks.config(text=str(self.detector.blink_counter))
        self.lbl_yawns.config(text=str(self.detector.yawn_counter))

        self._draw_sparkline(self.chart_ear, self.ear_history, COLORS["accent"])
        self._draw_sparkline(self.chart_mar, self.mar_history, "#B9C5E6")

        if self.detector.is_calibrated:
            self.lbl_calib.config(text="Hiệu chuẩn: Hoạt động ✓", fg=COLORS["safe"])
        else:
            rem = self.detector.CALIBRATION_FRAMES - len(self.detector.calibration_ear_buffer)
            self.lbl_calib.config(text=f"Hiệu chuẩn: Còn {rem} frames", fg=COLORS["warning"])

    # ═══════════════════════════════════════════════════════════════
    #  MAIN UPDATE LOOP
    # ═══════════════════════════════════════════════════════════════

    def update(self):
        if not self.running:
            return

        self.lbl_clock.config(text=datetime.now().strftime("%I:%M %p").lstrip("0"))

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Face detection
            if self.detector.use_mediapipe:
                faces = self.detector.detect_faces_mediapipe(frame)
            else:
                faces = self.detector.detect_faces_haar(frame)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            # Landmarks + metrics
            facial_metrics = None
            if faces and self.detector.use_facemesh:
                lm = self.detector.detect_face_landmarks(frame)
                if lm:
                    facial_metrics = self.detector.extract_facial_metrics(frame, lm)
                    self.detector.draw_facial_features(frame, facial_metrics)

            prediction_result = 3
            confidence = 0.0
            probabilities = self.current_probs

            if faces:
                x, y, w, h = faces[0]
                if w >= 50 and h >= 50:
                    try:
                        face_roi = frame[y: y + h, x: x + w]
                        face_input = self.detector.preprocess_face(face_roi)

                        inference_start = time.time()
                        preds = self.detector.model.predict(face_input, verbose=0)
                        inference_time = time.time() - inference_start
                        self.last_inference_ms = inference_time * 1000.0
                        self.detector.inference_times.append(inference_time)

                        smoothed = self.detector.smooth_predictions(preds[0])
                        p_class = int(np.argmax(smoothed))
                        conf = float(smoothed[p_class])

                        # Fuse + Stabilize (BUG FIX: old gui.py missed stabilize_class)
                        prediction_result, confidence = self.detector.fuse_prediction(
                            p_class, conf, facial_metrics
                        )
                        prediction_result, confidence = self.detector.stabilize_class(
                            prediction_result, confidence
                        )
                        probabilities = np.array(smoothed, dtype=np.float32)
                    except Exception:
                        prediction_result = 3
                        confidence = 0.0
                        probabilities = np.zeros(6, dtype=np.float32)
            else:
                probabilities = np.zeros(6, dtype=np.float32)

            self.current_prediction = prediction_result
            self.current_confidence = confidence
            self.current_probs = probabilities

            # Updates
            self._emit_realtime_events(len(faces), prediction_result, confidence, facial_metrics)
            self._draw_video_overlay(frame, faces, prediction_result, confidence, facial_metrics)
            self._update_status_display(prediction_result, facial_metrics, probabilities)
            self._update_metrics_display(facial_metrics)

            # Refresh alert timestamps periodically
            now = time.time()
            if now - self.last_alert_panel_refresh > 1.0:
                self._render_alert_cards()
                self.last_alert_panel_refresh = now

            # Render video to canvas
            img_h, img_w = frame.shape[:2]
            scale = min(self.canvas_width / img_w, self.canvas_height / img_h) if img_h > 0 else 1.0
            if scale > 0:
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                render = cv2.resize(frame, (new_w, new_h))
                render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(render))
                self.canvas.delete("all")
                self.canvas.create_image(
                    (self.canvas_width - new_w) // 2,
                    (self.canvas_height - new_h) // 2,
                    image=self.photo, anchor=tk.NW,
                )

            # FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            self.lbl_fps.config(text=f"FPS: {self.fps:.1f} | {self.last_inference_ms:.0f}ms")

        if self.running:
            self.window.after(self.delay, self.update)

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self.detector, "face_detector_task"):
            self.detector.face_detector_task.close()
        if hasattr(self.detector, "face_landmarker_task"):
            self.detector.face_landmarker_task.close()
        self.window.destroy()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = ModernDrowsinessApp(root, "AI Giám sát lái xe")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
