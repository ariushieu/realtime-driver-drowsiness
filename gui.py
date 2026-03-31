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

        self.colors = {
            "bg": "#0B1530",
            "sidebar": "#131E3A",
            "panel": "#1A2748",
            "panel_soft": "#202F54",
            "text_main": "#EEF5FF",
            "text_sub": "#95A8CB",
            "accent": "#35D8F6",
            "safe": "#37C978",
            "warning": "#F1B53F",
            "danger": "#E6576F",
            "drinking": "#3A9CFF",
            "border": "#2D416D",
            "bar_bg": "#2A3B64",
        }

        self.class_names_vi = {
            0: "Lái xe nguy hiểm",
            1: "Mất tập trung",
            2: "Đang uống nước",
            3: "Lái xe an toàn",
            4: "Buồn ngủ",
            5: "Ngáp",
        }

        self.class_colors = {
            0: self.colors["danger"],
            1: self.colors["warning"],
            2: self.colors["drinking"],
            3: self.colors["safe"],
            4: self.colors["danger"],
            5: self.colors["warning"],
        }

        self.window.configure(bg=self.colors["bg"])

        print("Đang khởi tạo bộ phát hiện...")
        self.detector = driver_detection.ImprovedDriverDetector(
            use_mediapipe=True,
            use_facemesh=True,
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        self.delay = 15
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_inference_ms = 0.0

        self.current_prediction = 3
        self.current_confidence = 0.0
        self.current_probs = np.zeros(6, dtype=np.float32)

        self.ear_history = deque(maxlen=40)
        self.mar_history = deque(maxlen=40)

        self.alert_events = deque(maxlen=8)
        self.last_event_key = None
        self.last_event_time = 0.0
        self.event_cooldowns = {
            "danger": 2.0,
            "distracted": 3.0,
            "drowsy": 2.0,
            "yawn": 4.0,
            "no_face": 6.0,
            "face_back": 4.0,
            "resolved": 8.0,
            "system": 12.0,
            "location": 20.0,
            "location_error": 30.0,
        }
        self.last_event_emit = {}
        self.no_face_start_time = None
        self.no_face_reported = False
        self.had_recent_warning = False
        self.last_alert_panel_refresh = 0.0

        self.prob_bars = []
        self.alert_widgets = []

        self.canvas_width = 900
        self.canvas_height = 520

        # Map widget state
        self.map_center_lat = 21.0285
        self.map_center_lon = 105.8542
        self.map_zoom = 13
        self.map_bg_cache = None
        self.map_bg_size = (0, 0)
        self.map_photo = None
        self.map_pulse_phase = 0
        self.map_last_refresh = 0.0

        # Location service state (IP-based geolocation)
        self.location_enabled = True
        self.location_refresh_interval = 180.0
        self.location_last_update = 0.0
        self.location_fetch_in_progress = False
        self.location_name = "Đang xác định vị trí..."

        self._setup_ui()
        self._seed_alerts()
        self._request_location_update(force=True)
        self._schedule_map_animation()
        self.update()

    def _setup_ui(self):
        self._build_sidebar()
        self._build_right_panel()
        self._build_main_area()

    def _build_sidebar(self):
        sidebar = tk.Frame(self.window, bg=self.colors["sidebar"], width=78)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        logo_canvas = tk.Canvas(sidebar, width=78, height=78, bg=self.colors["sidebar"], highlightthickness=0)
        logo_canvas.pack(pady=(8, 20))
        logo_canvas.create_oval(20, 18, 58, 56, fill=self.colors["panel_soft"], outline=self.colors["border"])
        logo_canvas.create_text(39, 37, text="*", fill=self.colors["accent"], font=("Consolas", 15, "bold"))

        items = [("U", False), ("C", True), ("M", False), ("S", False), ("A", False), ("D", False)]
        for symbol, active in items:
            row = tk.Frame(sidebar, bg=self.colors["accent"] if active else self.colors["sidebar"], height=46)
            row.pack(fill=tk.X, padx=10, pady=6)
            row.pack_propagate(False)
            tk.Label(
                row,
                text=symbol,
                font=("Segoe UI", 12, "bold"),
                fg=self.colors["sidebar"] if active else self.colors["text_sub"],
                bg=self.colors["accent"] if active else self.colors["sidebar"],
            ).pack(expand=True)

        footer = tk.Frame(sidebar, bg=self.colors["sidebar"])
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=14)
        avatar = tk.Canvas(footer, width=34, height=34, bg=self.colors["sidebar"], highlightthickness=0)
        avatar.pack()
        avatar.create_oval(3, 3, 31, 31, fill="#F1D0B5", outline=self.colors["border"])

    def _build_right_panel(self):
        self.right_panel = tk.Frame(self.window, bg=self.colors["panel"], width=360)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        self.right_panel.pack_propagate(False)

        pad = tk.Frame(self.right_panel, bg=self.colors["panel"], padx=18, pady=18)
        pad.pack(fill=tk.BOTH, expand=True)

        title_row = tk.Frame(pad, bg=self.colors["panel"])
        title_row.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            title_row,
            text="Cảnh báo & Sự kiện",
            font=("Segoe UI", 20, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT)
        tk.Label(
            title_row,
            text="Q",
            font=("Segoe UI", 11, "bold"),
            fg=self.colors["text_sub"],
            bg=self.colors["panel_soft"],
            padx=8,
            pady=3,
        ).pack(side=tk.RIGHT)

        self.alert_list = tk.Frame(pad, bg=self.colors["panel"])
        self.alert_list.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            pad,
            text="Xem toàn bộ nhật ký",
            font=("Segoe UI", 12, "bold"),
            fg="#F7FCFF",
            bg="#2E77FF",
            pady=10,
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=(14, 0))

    def _build_main_area(self):
        main_area = tk.Frame(self.window, bg=self.colors["bg"], padx=14, pady=14)
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Frame(main_area, bg=self.colors["bg"])
        header.pack(fill=tk.X, pady=(2, 10))

        title = tk.Frame(header, bg=self.colors["bg"])
        title.pack(side=tk.LEFT)
        tk.Label(
            title,
            text="AI GIÁM SÁT LÁI XE",
            font=("Segoe UI", 37, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["bg"],
        ).pack(anchor=tk.W)
        tk.Label(
            title,
            text="Hệ thống giám sát trạng thái tài xế thời gian thực",
            font=("Segoe UI", 14),
            fg=self.colors["text_sub"],
            bg=self.colors["bg"],
        ).pack(anchor=tk.W)

        status_center = tk.Frame(header, bg=self.colors["bg"])
        status_center.pack(side=tk.RIGHT)
        self.lbl_clock = tk.Label(
            status_center,
            text="10:11 AM",
            font=("Segoe UI", 26, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["bg"],
        )
        self.lbl_clock.pack(anchor=tk.E)
        tk.Label(
            status_center,
            text="Trực tiếp",
            font=("Segoe UI", 13),
            fg=self.colors["text_sub"],
            bg=self.colors["bg"],
        ).pack(anchor=tk.E)

        self.lbl_fps_sys = tk.Label(
            header,
            text="FPS hệ thống: 0.0",
            font=("Segoe UI", 18),
            fg=self.colors["safe"],
            bg=self.colors["bg"],
        )
        self.lbl_fps_sys.place(relx=0.62, rely=0.38, anchor="center")

        content = tk.Frame(main_area, bg=self.colors["bg"])
        content.pack(fill=tk.BOTH, expand=True)
        content.rowconfigure(0, weight=3)
        content.rowconfigure(1, weight=2)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        video_card = tk.Frame(content, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.canvas = tk.Canvas(video_card, bg="#08101F", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        diag_card = tk.Frame(content, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1, padx=14, pady=14)
        diag_card.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

        self.status_box = tk.Frame(diag_card, bg=self.colors["safe"], padx=12, pady=10)
        self.status_box.pack(fill=tk.X, pady=(0, 14))
        self.lbl_status_main = tk.Label(
            self.status_box,
            text="AN TOÀN",
            font=("Segoe UI", 48, "bold"),
            fg="#EFFCF4",
            bg=self.colors["safe"],
        )
        self.lbl_status_main.pack(anchor=tk.W)
        self.lbl_status_sub = tk.Label(
            self.status_box,
            text="Tài xế đang tập trung",
            font=("Segoe UI", 16),
            fg="#E8FFF0",
            bg=self.colors["safe"],
        )
        self.lbl_status_sub.pack(anchor=tk.W)

        self.probs_frame = tk.Frame(diag_card, bg=self.colors["panel"])
        self.probs_frame.pack(fill=tk.X)
        for cls_id in range(6):
            row = tk.Frame(self.probs_frame, bg=self.colors["panel"])
            row.pack(fill=tk.X, pady=2)

            bar = tk.Canvas(row, width=190, height=12, bg=self.colors["bar_bg"], highlightthickness=0)
            bar.pack(side=tk.LEFT)
            rect = bar.create_rectangle(0, 0, 0, 12, fill=self.class_colors[cls_id], width=0)

            label = tk.Label(
                row,
                text=f"{self.class_names_vi[cls_id]}: 0%",
                font=("Segoe UI", 11),
                fg=self.colors["text_main"],
                bg=self.colors["panel"],
                anchor="w",
            )
            label.pack(side=tk.LEFT, padx=(10, 0))

            self.prob_bars.append({"canvas": bar, "rect": rect, "label": label, "cls": cls_id})

        tk.Frame(diag_card, bg=self.colors["border"], height=1).pack(fill=tk.X, pady=10)
        tk.Label(
            diag_card,
            text="Chẩn đoán hệ thống:",
            font=("Segoe UI", 16, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["panel"],
        ).pack(anchor=tk.W)
        self.lbl_calib = tk.Label(
            diag_card,
            text="Hiệu chuẩn: Đang khởi tạo...",
            font=("Segoe UI", 12),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
        )
        self.lbl_calib.pack(anchor=tk.W)
        tk.Label(
            diag_card,
            text="Nguồn video: Camera 1",
            font=("Segoe UI", 12),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
        ).pack(anchor=tk.W)
        self.lbl_cam_fps = tk.Label(
            diag_card,
            text="FPS video: 0.0",
            font=("Segoe UI", 12),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
        )
        self.lbl_cam_fps.pack(anchor=tk.W)
        self.lbl_location_status = tk.Label(
            diag_card,
            text="Vị trí: Đang xác định...",
            font=("Segoe UI", 12),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
        )
        self.lbl_location_status.pack(anchor=tk.W)

        metrics_panel = tk.Frame(content, bg=self.colors["bg"])
        metrics_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        metrics_panel.rowconfigure(0, weight=1)
        metrics_panel.rowconfigure(1, weight=1)
        metrics_panel.columnconfigure(0, weight=1)
        metrics_panel.columnconfigure(1, weight=1)

        self.val_ear, self.chart_ear = self._create_metric_card(metrics_panel, "EAR (MẮT)", "0.00", 0, 0, "O")
        self.val_mar, self.chart_mar = self._create_metric_card(metrics_panel, "MAR (MIỆNG)", "0.00", 0, 1, "M")
        self.val_blinks, _ = self._create_metric_card(metrics_panel, "SỐ LẦN CHỚP MẮT", "0", 1, 0, "B")
        self.val_yawns, _ = self._create_metric_card(metrics_panel, "SỐ LẦN NGÁP", "0", 1, 1, "Y")

        map_card = tk.Frame(content, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1, padx=8, pady=8)
        map_card.grid(row=1, column=1, sticky="nsew")
        map_head = tk.Frame(map_card, bg=self.colors["panel"])
        map_head.pack(fill=tk.X, padx=4, pady=(2, 6))
        map_head_top = tk.Frame(map_head, bg=self.colors["panel"])
        map_head_top.pack(fill=tk.X)
        tk.Label(
            map_head_top,
            text="Bản đồ vị trí",
            font=("Segoe UI", 13, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT)
        tk.Button(
            map_head_top,
            text="Làm mới vị trí",
            command=lambda: self._request_location_update(force=True),
            font=("Segoe UI", 9, "bold"),
            fg=self.colors["text_main"],
            bg=self.colors["panel_soft"],
            activebackground=self.colors["accent"],
            activeforeground=self.colors["sidebar"],
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        map_head_bottom = tk.Frame(map_head, bg=self.colors["panel"])
        map_head_bottom.pack(fill=tk.X, pady=(4, 0))
        self.lbl_map_place = tk.Label(
            map_head_bottom,
            text=self.location_name,
            font=("Segoe UI", 10),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
            anchor="w",
        )
        self.lbl_map_place.pack(side=tk.LEFT)
        self.lbl_map_coord = tk.Label(
            map_head_bottom,
            text=f"{self.map_center_lat:.4f}, {self.map_center_lon:.4f}",
            font=("Consolas", 10),
            fg=self.colors["text_sub"],
            bg=self.colors["panel"],
        )
        self.lbl_map_coord.pack(side=tk.RIGHT)

        self.map_canvas = tk.Canvas(map_card, bg="#1A2748", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True)
        self.map_canvas.bind("<Configure>", self._draw_map)

    def _create_metric_card(self, parent, title, initial_value, row, col, icon):
        card = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1, padx=14, pady=12)
        card.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 6), pady=(0 if row == 0 else 6))

        head = tk.Frame(card, bg=self.colors["panel"])
        head.pack(fill=tk.X)
        tk.Label(head, text=title, font=("Segoe UI", 17, "bold"), fg=self.colors["text_sub"], bg=self.colors["panel"]).pack(side=tk.LEFT)
        tk.Label(head, text=icon, font=("Consolas", 14, "bold"), fg=self.colors["text_sub"], bg=self.colors["panel"]).pack(side=tk.RIGHT)

        val = tk.Label(card, text=initial_value, font=("Segoe UI", 44, "bold"), fg=self.colors["text_main"], bg=self.colors["panel"])
        val.pack(anchor=tk.W)

        chart = tk.Canvas(card, height=34, bg=self.colors["panel"], highlightthickness=0)
        chart.pack(fill=tk.X)
        return val, chart

    def _build_map_url(self, width, height):
        # Use Yandex static maps: no API key required for basic usage.
        size_w = max(200, min(width, 650))
        size_h = max(140, min(height, 450))
        query = urlencode(
            {
                "ll": f"{self.map_center_lon:.6f},{self.map_center_lat:.6f}",
                "z": str(self.map_zoom),
                "size": f"{size_w},{size_h}",
                "l": "map",
                "lang": "en_US",
            }
        )
        return f"https://static-maps.yandex.ru/1.x/?{query}"

    def _request_location_update(self, force=False):
        if not self.location_enabled or self.location_fetch_in_progress:
            return

        now = time.time()
        if not force and (now - self.location_last_update) < self.location_refresh_interval:
            return

        self.location_fetch_in_progress = True
        self.location_last_update = now

        worker = threading.Thread(target=self._location_lookup_worker, daemon=True)
        worker.start()

    def _location_lookup_worker(self):
        # IP-based geolocation (approximate), no extra dependency required.
        endpoints = [
            "https://ipapi.co/json/",
            "https://ipwho.is/",
        ]

        last_error = ""
        for endpoint in endpoints:
            try:
                with urlopen(endpoint, timeout=3.5) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="ignore"))

                lat = payload.get("latitude", payload.get("lat"))
                lon = payload.get("longitude", payload.get("lon"))
                if lat is None or lon is None:
                    raise ValueError("Không nhận được tọa độ")

                city = payload.get("city") or payload.get("district") or "Không rõ thành phố"
                region = payload.get("region") or payload.get("region_name") or ""
                country = payload.get("country_name") or payload.get("country") or ""

                location_payload = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "city": city,
                    "region": region,
                    "country": country,
                    "source": endpoint,
                }
                self.window.after(0, lambda p=location_payload: self._apply_location_update(p))
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

        city = payload.get("city", "")
        region = payload.get("region", "")
        country = payload.get("country", "")
        area_text = ", ".join([p for p in [city, region, country] if p])
        self.location_name = area_text if area_text else "Không rõ khu vực"

        self.lbl_map_place.config(text=self.location_name, fg=self.colors["safe"])
        self.lbl_map_coord.config(text=f"{self.map_center_lat:.5f}, {self.map_center_lon:.5f}")
        self.lbl_location_status.config(text=f"Vị trí: {self.location_name}", fg=self.colors["safe"])

        self._push_alert(
            f"Cập nhật vị trí: {self.location_name}",
            "Dịch vụ vị trí",
            "INFO",
            event_key="location",
        )
        self._draw_map()

    def _handle_location_error(self, message):
        self.location_fetch_in_progress = False
        self.lbl_map_place.config(text="Không kết nối được dịch vụ định vị", fg=self.colors["warning"])
        self.lbl_location_status.config(text="Vị trí: Ngoại tuyến (bản đồ dự phòng)", fg=self.colors["warning"])
        self._push_alert("Không truy cập được dịch vụ định vị", "Dịch vụ vị trí", "WARNING", event_key="location_error")

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

        # Draw stylized street blocks to mimic a map panel.
        for x in range(0, w + 1, max(28, w // 12)):
            self.map_canvas.create_line(x, 0, x, h, fill="#24385F", width=1)
        for y in range(0, h + 1, max(24, h // 10)):
            self.map_canvas.create_line(0, y, w, y, fill="#24385F", width=1)

        for i in range(0, 6):
            x0 = int((i * 0.17 + 0.05) * w)
            y0 = int((0.12 + (i % 3) * 0.22) * h)
            x1 = min(w - 8, x0 + int(0.14 * w))
            y1 = min(h - 8, y0 + int(0.13 * h))
            self.map_canvas.create_rectangle(x0, y0, x1, y1, outline="#2D416D", fill="#182845", width=1)

    def _draw_map_pin(self, w, h):
        cx, cy = w // 2, h // 2
        pulse = 10 + (self.map_pulse_phase % 8) * 2
        self.map_canvas.create_oval(
            cx - pulse,
            cy - pulse,
            cx + pulse,
            cy + pulse,
            outline="#35D8F6",
            width=2,
        )
        self.map_canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#35D8F6", outline="#D6FAFF", width=2)
        self.map_canvas.create_text(
            cx,
            min(h - 14, cy + 24),
            text="Vị trí xe",
            fill=self.colors["text_main"],
            font=("Segoe UI", 10, "bold"),
        )

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

    def _on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    def _seed_alerts(self):
        self._push_alert("Hệ thống giám sát đã khởi động", "Trợ lý AI", "INFO", event_key="system", force=True)
        self._push_alert("Đang kết nối Camera 1", "Trợ lý AI", "INFO", event_key="system", force=True)

    def _state_from_prediction(self, predicted_class, facial_metrics):
        is_drowsy = (
            facial_metrics
            and facial_metrics.get("is_drowsy")
            and self.detector.drowsy_frames > self.detector.DROWSY_FRAMES_THRESHOLD
        )

        state = {
            "key": "safe",
            "bg": self.colors["safe"],
            "title": "AN TOÀN",
            "subtitle": "Tài xế đang tập trung tốt",
            "badge": "RESOLVED",
        }

        if is_drowsy or predicted_class == 4:
            state = {
                "key": "drowsy",
                "bg": self.colors["danger"],
                "title": "BUỒN NGỦ",
                "subtitle": "Hãy tỉnh táo ngay",
                "badge": "CRITICAL",
            }
        elif predicted_class == 5:
            state = {
                "key": "yawn",
                "bg": self.colors["warning"],
                "title": "NGÁP",
                "subtitle": "Dấu hiệu mệt mỏi",
                "badge": "WARNING",
            }
        elif predicted_class == 1:
            state = {
                "key": "distracted",
                "bg": self.colors["warning"],
                "title": "MẤT TẬP TRUNG",
                "subtitle": "Hãy nhìn đường",
                "badge": "WARNING",
            }
        elif predicted_class == 0:
            state = {
                "key": "danger",
                "bg": self.colors["danger"],
                "title": "NGUY HIỂM",
                "subtitle": "Hành vi lái xe bất thường",
                "badge": "CRITICAL",
            }
        elif predicted_class == 2:
            state = {
                "key": "drinking",
                "bg": self.colors["drinking"],
                "title": "ĐANG UỐNG",
                "subtitle": "Đã phát hiện hành động",
                "badge": "INFO",
            }

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

        badge_text = {
            "CRITICAL": "NGHIÊM TRỌNG",
            "WARNING": "CẢNH BÁO",
            "RESOLVED": "ĐÃ ỔN ĐỊNH",
            "INFO": "THÔNG TIN",
        }.get(badge, badge)

        badge_color = {
            "CRITICAL": self.colors["danger"],
            "WARNING": self.colors["warning"],
            "RESOLVED": self.colors["safe"],
            "INFO": self.colors["drinking"],
        }.get(badge, self.colors["text_sub"])

        self.alert_events.appendleft(
            {
                "title": title,
                "actor": actor,
                "clock": datetime.now().strftime("%I:%M %p").lstrip("0"),
                "created_at": now,
                "badge": badge,
                "badge_text": badge_text,
                "badge_color": badge_color,
                "event_key": event_key or "generic",
            }
        )
        self._render_alert_cards()

    def _render_alert_cards(self):
        for w in self.alert_widgets:
            w.destroy()
        self.alert_widgets = []

        for event in list(self.alert_events)[:4]:
            card = tk.Frame(
                self.alert_list,
                bg=self.colors["panel_soft"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                padx=12,
                pady=10,
            )
            card.pack(fill=tk.X, pady=(0, 12))

            row_top = tk.Frame(card, bg=self.colors["panel_soft"])
            row_top.pack(fill=tk.X)
            avatar = tk.Canvas(row_top, width=34, height=34, bg=self.colors["panel_soft"], highlightthickness=0)
            avatar.pack(side=tk.LEFT, padx=(0, 10))
            avatar.create_oval(3, 3, 31, 31, fill="#D8BA9A", outline="#7D8FBA")
            tk.Label(
                row_top,
                text=event["title"],
                font=("Segoe UI", 15, "bold"),
                fg=self.colors["text_main"],
                bg=self.colors["panel_soft"],
                anchor="w",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            row_bot = tk.Frame(card, bg=self.colors["panel_soft"])
            row_bot.pack(fill=tk.X, pady=(6, 0))
            left = tk.Frame(row_bot, bg=self.colors["panel_soft"])
            left.pack(side=tk.LEFT)
            tk.Label(left, text=event["actor"], font=("Segoe UI", 12, "bold"), fg=self.colors["text_main"], bg=self.colors["panel_soft"]).pack(anchor=tk.W)
            relative_time = self._format_relative_time(event.get("created_at", time.time()))
            tk.Label(
                left,
                text=f"{event.get('clock', '--:--')} | {relative_time}",
                font=("Segoe UI", 11),
                fg=self.colors["text_sub"],
                bg=self.colors["panel_soft"],
            ).pack(anchor=tk.W)

            tk.Label(
                row_bot,
                text=event.get("badge_text", event["badge"]),
                font=("Segoe UI", 11, "bold"),
                fg="#F8FBFF",
                bg=event["badge_color"],
                padx=10,
                pady=4,
            ).pack(side=tk.RIGHT)

            self.alert_widgets.append(card)

    def _draw_sparkline(self, canvas, values, color):
        width = max(canvas.winfo_width(), 180)
        height = max(canvas.winfo_height(), 30)
        canvas.delete("all")

        if len(values) < 2:
            return

        arr = np.array(values, dtype=np.float32)
        min_v = float(arr.min())
        max_v = float(arr.max())
        span = max(max_v - min_v, 1e-6)

        points = []
        for i, val in enumerate(arr):
            x = int(i * (width - 4) / (len(arr) - 1)) + 2
            y = int((1.0 - (val - min_v) / span) * (height - 8)) + 4
            points.extend([x, y])

        canvas.create_line(points, fill=color, width=2, smooth=True)

    def _update_probability_bars(self, probabilities):
        probs = probabilities if probabilities is not None else np.zeros(6, dtype=np.float32)
        for i, item in enumerate(self.prob_bars):
            p = float(probs[i]) if i < len(probs) else 0.0
            p = min(max(p, 0.0), 1.0)
            width = int(item["canvas"].winfo_width())
            if width <= 1:
                width = 190
            item["canvas"].coords(item["rect"], 0, 0, int(width * p), 12)
            item["label"].config(text=f"{self.class_names_vi[i]}: {p:.0%}")

    def _to_cv_text(self, text):
        raw = str(text)
        # OpenCV Hershey fonts do not support Vietnamese accents reliably.
        safe = raw.replace("Đ", "D").replace("đ", "d")
        safe = unicodedata.normalize("NFKD", safe).encode("ascii", "ignore").decode("ascii")
        return safe if safe else raw

    def _draw_video_overlay(self, frame, faces, predicted_class, confidence, facial_metrics):
        h, w = frame.shape[:2]
        color_hex = self.class_colors.get(predicted_class, self.colors["safe"])
        bgr = tuple(int(color_hex[i : i + 2], 16) for i in (5, 3, 1))

        if faces:
            x, y, fw, fh = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), bgr, 2)
            label = f"{self.class_names_vi.get(predicted_class, 'Không rõ')} ({confidence:.0%})"
            cv2.putText(frame, self._to_cv_text(label), (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)

        cv2.putText(frame, self._to_cv_text(f"FPS: {self.fps:.1f}"), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (62, 255, 174), 2)
        cv2.putText(frame, self._to_cv_text(f"Tốc độ: {self.last_inference_ms:.0f}ms"), (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (62, 220, 190), 2)

        if facial_metrics:
            cv2.putText(frame, self._to_cv_text(f"Ngáp: {self.detector.yawn_counter}"), (12, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 230, 220), 2)
            cv2.putText(frame, self._to_cv_text(f"Miệng ({facial_metrics['mar']:.2f})"), (12, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 230, 220), 2)

        method = "MediaPipe" if self.detector.use_mediapipe else "Haar"
        cv2.putText(frame, self._to_cv_text(method), (w - 170, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (225, 245, 255), 2)

    def _update_status_display(self, predicted_class, facial_metrics, probabilities):
        state = self._state_from_prediction(predicted_class, facial_metrics)

        self.status_box.configure(bg=state["bg"])
        self.lbl_status_main.configure(text=state["title"], bg=state["bg"])
        self.lbl_status_sub.configure(text=state["subtitle"], bg=state["bg"])
        self._update_probability_bars(probabilities)

    def _emit_realtime_events(self, face_count, predicted_class, confidence, facial_metrics):
        now = time.time()

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
            facial_metrics
            and facial_metrics.get("is_drowsy")
            and self.detector.drowsy_frames > max(6, self.detector.DROWSY_FRAMES_THRESHOLD // 2)
        )
        is_yawning = bool(facial_metrics and facial_metrics.get("is_yawning"))

        if predicted_class == 0 and confidence >= 0.35:
            self._push_alert("Lái xe nguy hiểm", "Trợ lý AI", "CRITICAL", event_key="danger")
            self.had_recent_warning = True

        if predicted_class == 1 and confidence >= 0.35:
            self._push_alert("Mất tập trung", "Trợ lý AI", "WARNING", event_key="distracted")
            self.had_recent_warning = True

        if predicted_class == 4 or is_drowsy:
            self._push_alert(
                f"Buồn ngủ ({self.detector.drowsy_frames} khung hình)",
                "Trợ lý AI",
                "CRITICAL",
                event_key="drowsy",
            )
            self.had_recent_warning = True

        if predicted_class == 5 or is_yawning:
            self._push_alert(
                f"Phát hiện ngáp (tổng: {self.detector.yawn_counter})",
                "Trợ lý AI",
                "WARNING",
                event_key="yawn",
            )
            self.had_recent_warning = True

        is_safe_state = face_count > 0 and predicted_class == 3 and not is_drowsy and not is_yawning
        if is_safe_state and self.had_recent_warning:
            self._push_alert("Trạng thái đã ổn định", "Trợ lý AI", "RESOLVED", event_key="resolved")
            self.had_recent_warning = False

    def _update_metrics_display(self, facial_metrics):
        if facial_metrics:
            ear = float(facial_metrics["ear"])
            mar = float(facial_metrics["mar"])

            self.val_ear.config(text=f"{ear:.2f}")
            self.val_mar.config(text=f"{mar:.2f}")

            self.ear_history.append(ear)
            self.mar_history.append(mar)
        else:
            self.val_ear.config(text="--")
            self.val_mar.config(text="--")

        self.val_blinks.config(text=str(self.detector.blink_counter))
        self.val_yawns.config(text=str(self.detector.yawn_counter))

        self._draw_sparkline(self.chart_ear, self.ear_history, self.colors["accent"])
        self._draw_sparkline(self.chart_mar, self.mar_history, "#B9C5E6")

        if self.detector.is_calibrated:
            self.lbl_calib.config(text="Hiệu chuẩn: Đang hoạt động", fg=self.colors["safe"])
        else:
            rem = self.detector.CALIBRATION_FRAMES - len(self.detector.calibration_ear_buffer)
            self.lbl_calib.config(text=f"Hiệu chuẩn (Còn {rem} khung hình)", fg=self.colors["warning"])

    def update(self):
        if not self.running:
            return

        self.lbl_clock.config(text=datetime.now().strftime("%I:%M %p").lstrip("0"))

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            faces = self.detector.detect_faces_mediapipe(frame) if self.detector.use_mediapipe else self.detector.detect_faces_haar(frame)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

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
                        face_roi = frame[y : y + h, x : x + w]
                        face_input = self.detector.preprocess_face(face_roi)

                        inference_start = time.time()
                        preds = self.detector.model.predict(face_input, verbose=0)
                        inference_time = time.time() - inference_start
                        self.last_inference_ms = inference_time * 1000.0
                        self.detector.inference_times.append(inference_time)

                        smoothed = self.detector.smooth_predictions(preds[0])
                        p_class = int(np.argmax(smoothed))
                        conf = float(smoothed[p_class])
                        prediction_result, confidence = self.detector.fuse_prediction(p_class, conf, facial_metrics)
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

            self._emit_realtime_events(len(faces), prediction_result, confidence, facial_metrics)
            self._draw_video_overlay(frame, faces, prediction_result, confidence, facial_metrics)
            self._update_status_display(prediction_result, facial_metrics, probabilities)
            self._update_metrics_display(facial_metrics)

            now = time.time()
            if now - self.last_alert_panel_refresh > 1.0:
                self._render_alert_cards()
                self.last_alert_panel_refresh = now

            img_h, img_w = frame.shape[:2]
            scale = min(self.canvas_width / img_w, self.canvas_height / img_h) if img_h > 0 else 1.0
            if scale > 0:
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                render = cv2.resize(frame, (new_w, new_h))
                render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(render))

                self.canvas.delete("all")
                self.canvas.create_image((self.canvas_width - new_w) // 2, (self.canvas_height - new_h) // 2, image=self.photo, anchor=tk.NW)

            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            self.lbl_fps_sys.config(text=f"FPS hệ thống: {self.fps:.1f}")
            self.lbl_cam_fps.config(text=f"FPS video: {self.fps:.1f}")

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


if __name__ == "__main__":
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    app = ModernDrowsinessApp(root, "AI Giám sát lái xe - Bảng điều khiển")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
