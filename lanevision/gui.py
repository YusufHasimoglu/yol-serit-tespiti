"""Tkinter ile Türkçe masaüstü arayüz. OpenCV worker yalnızca kuyruk üzerinden UI'yi günceller."""
from __future__ import annotations

import queue
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .core import Detection, LaneSettings
from .engine import InferenceEngine, MODES
from .feedback import (find_saved_review, render_reviewed_annotation,
                       save_confirmed_detection, style_sample_count)
from .live_video import LiveVideoSession
from .video_feedback import load_video_anchors, save_video_anchor
from .video_training import video_training_stats
from .ufld import DEFAULT_UFLD_WEIGHTS
from .yolopv2 import DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS, DEFAULT_YOLOPV2_WEIGHTS
from .processing import VIDEO_EXTENSIONS, process_video, read_image, save_stages, write_image
from .scene import analyze_scene

APP_ROOT = Path(__file__).resolve().parent.parent
BG = "#0B1220"
SIDEBAR = "#0E1726"
PANEL = "#162235"
BLUE = "#3D8DCC"
TEXT = "#F1F5F9"
MUTED = "#9EADBD"
GOOD = "#35AE71"
BORDER = "#2A3C53"
PANEL_DARK = "#080F1B"
PANEL_HOVER = "#20344D"
WARM = "#D9A545"
DANGER = "#B9484E"
# Teknik modlar test/komut satırı için korunur. Masaüstü arayüzü ise günlük
# kullanımda yalnızca anlaşılır üç seçeneği gösterir.
USER_VISIBLE_MODES={
    "auto":"Otomatik (görsele göre önerilen)",
    "yolopv2":"Gelişmiş model analizi",
    "classic":"Klasik çizgi takibi (hızlı)",
}
STAGE_LABELS={
    "01 Orijinal": "Özgün görüntü",
    "02 Gri ton": "Gri ton",
    "03 Bulaniklastirma": "Yumuşatılmış görünüm",
    "04 Canny kenarlar": "Kenar haritası",
    "05 ROI maskesi": "İnceleme alanı",
    "06 ROI kenarlari": "İnceleme alanı kenarları",
    "07 Hough dogrulari": "Çizgi adayları",
    "08 Sonuc": "Sonuç görünümü",
    "09 AI olasilik haritasi": "Model güven haritası",
    "10 AI sinif maskeleri": "Çizgi sınıfları",
    "11 Kullanıcı düzeltmesi": "Kullanıcı doğrulaması",
}


class LaneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ŞERİT ANALİZİ  |  Yol Analiz Merkezi")
        # Windows görüntü ölçeklendirmesinde (özellikle %125) alt eylem çubuğu
        # ekran dışına taşmasın diye pencereyi kullanılabilir ekrana göre kur.
        width = min(1440, max(1050, self.winfo_screenwidth() - 72))
        height = min(780, max(660, self.winfo_screenheight() - 86))
        self.geometry(f"{width}x{height}")
        self.minsize(1050, 660)
        self.configure(bg=BG)
        self.current_file: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_detection: Detection | None = None
        self.current_display: np.ndarray | None = None
        self.output_video: Path | None = None
        self.is_processing = False
        self.live_session: LiveVideoSession | None = None
        self.live_playing = False
        self.live_after_id = None
        self.live_frame_index = 0
        self.pending_manual_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self.annotation_in_progress = False
        self.video_total_frames = 0
        self.video_selected_index = 0
        self.video_seek_after_id = None
        self.video_seek_sync = False
        self.stop_event = threading.Event()
        self.messages: queue.Queue[tuple] = queue.Queue(maxsize=80)
        self.preview_photo = None
        self.settings_vars = {
            "canny_low": tk.IntVar(value=55),
            "canny_high": tk.IntVar(value=160),
            "roi_top": tk.IntVar(value=59),
            "hough_threshold": tk.IntVar(value=28),
            "min_line_length": tk.IntVar(value=24),
            "max_line_gap": tk.IntVar(value=36),
            "smoothing": tk.IntVar(value=60),
        }
        self.selected_stage = tk.StringVar(value="08 Sonuc")
        self.stage_label = tk.StringVar(value=STAGE_LABELS["08 Sonuc"])
        self.mode_var = tk.StringVar(value=USER_VISIBLE_MODES["auto"])
        self.mode_keys = {label:key for key,label in USER_VISIBLE_MODES.items()}
        self.algorithm_method = tk.StringVar(value="Çalışma profili: beklemede")
        self.feedback_status = tk.StringVar(value="Sonucu onaylayın veya gerekirse sınırları düzeltin.")
        self.status = tk.StringVar(value="Hazır. Örnek görüntüyle başlayabilir veya kendi dosyanızı açabilirsiniz.")
        self.metric_left = tk.StringVar(value="—")
        self.metric_right = tk.StringVar(value="—")
        self.metric_offset = tk.StringVar(value="—")
        self.metric_segments = tk.StringVar(value="—")
        self.source_name = tk.StringVar(value="Dosya seçilmedi")
        self.video_frame_var = tk.IntVar(value=0)
        self.video_frame_text = tk.StringVar(value="Video karesi: video açın")
        self.scene_lighting = tk.StringVar(value="IŞIK: BEKLENİYOR")
        self.scene_geometry = tk.StringVar(value="YOL: BEKLENİYOR")
        self.scene_brightness = tk.StringVar(value="Görsel analizi bekleniyor")
        self.scene_profile = None
        self.ai_status_var = tk.StringVar(value="Analiz altyapısı denetleniyor")
        self.dashboard_detection = tk.StringVar(value="Analiz bekleniyor")
        self.dashboard_confidence = tk.StringVar(value="—")
        self.dashboard_fps = tk.StringVar(value="Hazır")
        self.dashboard_alert = tk.StringVar(value="Bir görüntü veya video seçin.")
        self.dashboard_latency = tk.StringVar(value="—")
        self.dashboard_stability = tk.StringVar(value="—")
        self.dashboard_curve = tk.StringVar(value="Belirsiz")
        self.dashboard_lane_count = tk.StringVar(value="0")
        self.dashboard_runtime = tk.StringVar(value="00:00:00")
        self.video_training_var = tk.StringVar(value="Video eğitimi hazırlanıyor")
        self.video_training_detail = tk.StringVar(value="Doğrulanmış video karesi bekleniyor")
        self.model_badge = tk.StringVar(value="Profil: hazırlanıyor")
        self.compute_badge = tk.StringVar(value=f"CPU: {os.cpu_count() or 1} çekirdek")
        self._activity_lines: list[str] = []
        self._offset_history: list[float] = []
        self._started_at = time.monotonic()
        self.auto_detect_var = tk.BooleanVar(value=True)
        self.advanced_visible = False
        self.progress_percent = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="İşlem bekleniyor")
        self._configure_styles()
        self._check_ai_ready()
        self._build_ui()
        self.status.trace_add("write", self._record_status)
        self._record_status()
        self._refresh_video_training_status()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._process_messages)
        self.after(200, lambda: self._load_example_if_exists(force=False))
        self.after(1000, self._tick_dashboard)

    def _check_ai_ready(self):
        import importlib.util
        from .ai_model import DEFAULT_WEIGHTS
        ufld_ready=DEFAULT_UFLD_WEIGHTS.is_file()
        yolopv2_ready=DEFAULT_YOLOPV2_WEIGHTS.is_file()
        feedback_ready=DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS.is_file()
        if importlib.util.find_spec("torch") is None:
            self.ai_status_var.set("• Analiz altyapısı kullanılamıyor")
            self.model_badge.set("Profil: kullanılamıyor")
        else:
            profiles_ready = any((DEFAULT_WEIGHTS.is_file(), ufld_ready, yolopv2_ready))
            if profiles_ready:
                self.ai_status_var.set("• Analiz altyapısı hazır")
                self.model_badge.set("Profil: gelişmiş" if yolopv2_ready else "Profil: standart")
            else:
                self.ai_status_var.set("• Analiz profili bekleniyor")
                self.model_badge.set("Profil: yapılandırılmadı")

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT,
                        selectforeground=TEXT, selectbackground=PANEL_HOVER, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, arrowcolor=GOOD, padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)],
                  foreground=[("readonly", TEXT)], background=[("readonly", PANEL)])
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL_DARK, background=GOOD,
                        bordercolor=PANEL_DARK, lightcolor=GOOD, darkcolor=GOOD, borderwidth=0)

    def _build_ui(self):
        shell = tk.Frame(self, bg=BG)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg=SIDEBAR, width=246, highlightbackground=BORDER, highlightthickness=1)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        side_brand = tk.Frame(sidebar, bg=SIDEBAR)
        side_brand.pack(fill="x", padx=22, pady=(25, 22))
        tk.Label(side_brand, text="◈", bg=SIDEBAR, fg=BLUE,
                 font=("Segoe UI Symbol", 22, "bold")).pack(side="left", padx=(0, 10))
        brand_text = tk.Frame(side_brand, bg=SIDEBAR)
        brand_text.pack(side="left")
        tk.Label(brand_text, text="Şerit Analizi", bg=SIDEBAR, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(brand_text, text="YOL GÖRÜNTÜSÜ İNCELEME", bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", pady=(2, 0))

        nav = tk.Frame(sidebar, bg=SIDEBAR)
        nav.pack(fill="x", padx=13)
        self._nav_buttons = []
        self._nav_button(nav, "⌂  Genel Bakış", lambda: self.image_panel.focus_set(), active=True)
        self._nav_button(nav, "▣  Görüntü", self._select_image)
        self._nav_button(nav, "▷  Video İncelemesi", self._select_video)
        self._nav_button(nav, "✎  Doğrulama", self._label_current_image)
        self._nav_button(nav, "▤  Kayıtlar", self._show_video_training_info)
        self._nav_button(nav, "▥  Raporlar", self._save_steps)
        self._nav_button(nav, "⚙  Ayarlar", self._toggle_advanced)
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=18)

        quick = tk.Frame(sidebar, bg=SIDEBAR)
        quick.pack(fill="x", padx=20)
        tk.Label(quick, text="HIZLI İŞLEMLER", bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 8))
        self._button(quick, "ÖRNEK ÇALIŞMA", lambda: self._load_example_if_exists(force=True), compact=True)
        self._button(quick, "DOĞRULAMA VERİSİNİ İŞLE", self._train_feedback_model, compact=True)
        training_side = tk.Frame(sidebar, bg=PANEL, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
        training_side.pack(fill="x", padx=15, pady=(15, 0))
        tk.Label(training_side, text="▣  VİDEO DOĞRULAMA KAYDI", bg=PANEL, fg=BLUE,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(training_side, textvariable=self.video_training_var, bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9, "bold"), wraplength=190, justify="left").pack(anchor="w", pady=(4, 0))
        tk.Label(training_side, textvariable=self.video_training_detail, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8), wraplength=190, justify="left").pack(anchor="w", pady=(2, 0))
        side_info = tk.Frame(sidebar, bg=PANEL, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
        side_info.pack(side="bottom", fill="x", padx=15, pady=17)
        tk.Label(side_info, text="●  ANALİZ ALTYAPISI", bg=PANEL, fg=GOOD,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(side_info, textvariable=self.ai_status_var, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8), wraplength=190, justify="left").pack(anchor="w", pady=(4, 0))

        app = tk.Frame(shell, bg=BG)
        app.pack(side="right", fill="both", expand=True)
        topbar = tk.Frame(app, bg=SIDEBAR, height=84, highlightbackground=BORDER, highlightthickness=1)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        top_title = tk.Frame(topbar, bg=SIDEBAR)
        top_title.pack(side="left", padx=26, pady=10)
        tk.Label(top_title, text="◈", bg=SIDEBAR, fg=BLUE,
                 font=("Segoe UI Symbol", 22, "bold")).pack(side="left", padx=(0, 12))
        title_text = tk.Frame(top_title, bg=SIDEBAR)
        title_text.pack(side="left")
        tk.Label(title_text, text="Yol Analiz Merkezi", bg=SIDEBAR, fg=TEXT,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(title_text, text="Görüntü, video ve kalite kontrolü çalışma alanı", bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")
        badges = tk.Frame(topbar, bg=SIDEBAR)
        badges.pack(side="right", padx=20, pady=15)
        self._top_badge(badges, "●  Sistem hazır", GOOD)
        self._top_badge(badges, self.model_badge, BLUE)
        self._top_badge(badges, self.compute_badge, MUTED)
        self._top_badge(badges, "⌁  Çalışma: yerel", GOOD)

        content = tk.Frame(app, bg=BG)
        content.pack(fill="both", expand=True, padx=18, pady=18)
        rail = tk.Frame(content, bg=BG, width=300)
        rail.pack(side="right", fill="y", padx=(16, 0))
        rail.pack_propagate(False)
        work = tk.Frame(content, bg=BG)
        work.pack(side="left", fill="both", expand=True)

        source_card = tk.Frame(work, bg=PANEL, padx=15, pady=8, highlightbackground=BORDER, highlightthickness=1)
        source_card.pack(fill="x", pady=(0, 12))
        source_head = tk.Frame(source_card, bg=PANEL)
        source_head.pack(fill="x")
        tk.Label(source_head, text="▣  Kaynak ve İnceleme Ayarları", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(source_head, textvariable=self.source_name, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), anchor="e").pack(side="right")
        source_row = tk.Frame(source_card, bg=PANEL)
        source_row.pack(fill="x", pady=(6, 2))
        self._button(source_row, "GÖRÜNTÜ AÇ", self._select_image, primary=True, compact=True,
                     pack=False).pack(side="left", padx=(0, 7))
        self._button(source_row, "VİDEO AÇ", self._select_video, compact=True,
                     pack=False).pack(side="left", padx=(0, 12))
        tk.Checkbutton(source_row, text="Otomatik analiz", variable=self.auto_detect_var,
                       bg=PANEL, fg=TEXT, selectcolor=PANEL_DARK, activebackground=PANEL,
                       activeforeground=TEXT, font=("Segoe UI", 9)).pack(side="left", padx=(0, 10))
        ttk.Combobox(source_row, textvariable=self.mode_var, values=list(USER_VISIBLE_MODES.values()),
                     state="readonly", width=30).pack(side="left", fill="x", expand=True)
        self.advanced_btn = self._button(source_row, "⚙ Ayarlar", self._toggle_advanced,
                                         compact=True, pack=False)
        self.advanced_btn.pack(side="right", padx=(10, 0))
        self.advanced_frame = tk.Frame(source_card, bg=PANEL)
        for index, (label, key, low, high) in enumerate((
            ("Canny alt eşik", "canny_low", 0, 220),
            ("Canny üst eşik", "canny_high", 1, 255),
            ("ROI başlangıcı (%)", "roi_top", 35, 80),
            ("Hough eşik", "hough_threshold", 5, 90),
            ("En küçük çizgi (px)", "min_line_length", 10, 140),
            ("Birleştirme aralığı", "max_line_gap", 5, 120),
            ("Video yumuşatma (%)", "smoothing", 0, 95),
        )):
            if index:
                tk.Frame(self.advanced_frame, bg=BORDER, height=1).pack(fill="x", pady=2)
            self._slider(self.advanced_frame, label, key, low, high)
        video_bar = tk.Frame(source_card, bg=PANEL)
        video_bar.pack(fill="x", pady=(2, 0))
        tk.Label(video_bar, textvariable=self.video_frame_text, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left")
        self.video_seek = tk.Scale(video_bar, from_=0, to=1, orient="horizontal", showvalue=False,
                                   variable=self.video_frame_var, command=self._queue_video_seek,
                                   state="disabled", bg=PANEL, fg=TEXT, troughcolor=PANEL_DARK,
                                   highlightthickness=0, activebackground=BLUE, bd=0, sliderlength=13)
        self.video_seek.pack(side="right", fill="x", expand=True, padx=(14, 0))

        visual = tk.Frame(work, bg=PANEL, padx=12, pady=11, height=265,
                          highlightbackground=BORDER, highlightthickness=1)
        visual.pack(fill="both", expand=True)
        # Görselin doğal piksel boyutu, alt kontrol kartlarını ekrandan itmemeli.
        visual.pack_propagate(False)
        visual_head = tk.Frame(visual, bg=PANEL)
        visual_head.pack(fill="x", pady=(0, 8))
        tk.Label(visual_head, text="◉  Görüntü İncelemesi", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        self.stage_combo = ttk.Combobox(visual_head, textvariable=self.stage_label, state="readonly", width=24,
                     values=[STAGE_LABELS[stage] for stage in STAGE_LABELS if stage != "11 Kullanıcı düzeltmesi"])
        self.stage_combo.pack(side="right")
        self.selected_stage.trace_add("write", self._stage_changed)
        self.stage_label.trace_add("write", self._stage_label_changed)
        self.image_panel = tk.Label(visual, bg=PANEL_DARK, fg=MUTED,
                                    text="Bir fotoğraf veya video açın", font=("Segoe UI", 13))
        self.image_panel.pack(fill="both", expand=True)
        self.image_panel.bind("<Configure>", lambda *_: self._render_preview())
        overlay = tk.Frame(visual, bg="#0D1827", padx=10, pady=7, highlightbackground=BORDER, highlightthickness=1)
        overlay.place(relx=.018, rely=.18, anchor="nw")
        tk.Label(overlay, text="◈  ŞERİT İNCELEMESİ", bg="#0D1827", fg=GOOD,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(overlay, textvariable=self.scene_lighting, bg="#0D1827", fg=TEXT,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))
        tk.Label(overlay, textvariable=self.scene_geometry, bg="#0D1827", fg=TEXT,
                 font=("Segoe UI", 8)).pack(anchor="w")
        tk.Label(overlay, textvariable=self.dashboard_curve, bg="#0D1827", fg=WARM,
                 font=("Segoe UI", 8)).pack(anchor="w")
        live_badge = tk.Label(visual, textvariable=self.dashboard_detection, bg="#183A2A", fg=GOOD,
                              padx=10, pady=6, font=("Segoe UI", 9, "bold"),
                              highlightbackground=GOOD, highlightthickness=1)
        live_badge.place(relx=.975, rely=.18, anchor="ne")

        tabs = tk.Frame(work, bg=BG)
        tabs.pack(fill="x", pady=(8, 0))
        self._stage_tab_buttons = {}
        self._stage_tab(tabs, "▣  Sonuç Görünümü", "08 Sonuc")
        self._stage_tab(tabs, "⠿  Kenar Haritası", "04 Canny kenarlar")
        self._stage_tab(tabs, "╱╲  Şerit Maskesi", "05 ROI maskesi")
        self._stage_tab(tabs, "◉  Model Görünümü", "09 AI olasilik haritasi")

        lower = tk.Frame(work, bg=BG)
        lower.pack(fill="x", pady=(12, 0))
        lower.grid_columnconfigure(0, weight=6)
        lower.grid_columnconfigure(1, weight=4)
        feedback = tk.Frame(lower, bg=PANEL, padx=13, pady=10, highlightbackground=BORDER, highlightthickness=1)
        feedback.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        tk.Label(feedback, text="✎  Kalite Kontrolü", bg=PANEL, fg=BLUE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(feedback, textvariable=self.feedback_status, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8), anchor="w", justify="left", wraplength=400).pack(fill="x", pady=(3, 7))
        feedback_actions = tk.Frame(feedback, bg=PANEL)
        feedback_actions.pack(fill="x")
        for column in range(3):
            feedback_actions.grid_columnconfigure(column, weight=1)
        self.btn_feedback_ok = tk.Button(feedback_actions, text="✓ ONAYLA",
                                         command=self._confirm_detection, state="disabled",
                                         bg=GOOD, fg="#06130D", activebackground="#7BFF9E", bd=0,
                                         font=("Segoe UI", 8, "bold"), padx=7, pady=5, cursor="hand2")
        self.btn_feedback_ok.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.btn_feedback_fix = tk.Button(feedback_actions, text="✎ DÜZELT",
                                          command=self._review_incorrect, state="disabled",
                                          bg=PANEL_HOVER, fg=TEXT, activebackground="#294C77", bd=0,
                                          font=("Segoe UI", 8, "bold"), padx=7, pady=5, cursor="hand2")
        self.btn_feedback_fix.grid(row=0, column=1, sticky="ew", padx=3)
        self.btn_feedback_train = tk.Button(feedback_actions, text="VERİYİ İŞLE",
                                            command=self._train_feedback_model,
                                            bg=BLUE, fg="#04111F", activebackground="#7ACCFF", bd=0,
                                            font=("Segoe UI", 8, "bold"), padx=7, pady=5, cursor="hand2")
        self.btn_feedback_train.grid(row=0, column=2, sticky="ew", padx=(3, 0))
        log_card = tk.Frame(lower, bg=PANEL, padx=13, pady=10, highlightbackground=BORDER, highlightthickness=1)
        log_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        tk.Label(log_card, text="▤  Sistem Mesajları", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.activity_log = tk.Text(log_card, height=3, bg=PANEL_DARK, fg=MUTED, bd=0,
                                    highlightthickness=0, font=("Consolas", 8), state="disabled",
                                    wrap="word", padx=8, pady=6)
        self.activity_log.pack(fill="both", expand=True, pady=(6, 0))

        action_card = tk.Frame(work, bg=SIDEBAR, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
        action_card.pack(fill="x", pady=(12, 0))
        self.progressbar = ttk.Progressbar(action_card, variable=self.progress_percent, maximum=100,
                                           style="Horizontal.TProgressbar")
        self.progressbar.pack(fill="x")
        tk.Label(action_card, textvariable=self.progress_text, bg=SIDEBAR, fg=MUTED, anchor="w",
                 font=("Segoe UI", 8)).pack(fill="x", pady=(3, 7))
        action_row = tk.Frame(action_card, bg=SIDEBAR)
        action_row.pack(fill="x")
        for column in range(6):
            action_row.grid_columnconfigure(column, weight=1)
        self.btn_live = self._button(action_row, "▣ Canlı İnceleme", self._start_live_video, compact=True, pack=False)
        self.btn_live.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_live.configure(state="disabled")
        self.btn_detect = self._button(action_row, "▶ İncelemeyi Başlat", self._detect, primary=True, compact=True, pack=False)
        self.btn_detect.grid(row=0, column=1, sticky="ew", padx=5)
        self.btn_label = self._button(action_row, "✎ Etiketle", self._label_current_image, compact=True, pack=False)
        self.btn_label.grid(row=0, column=2, sticky="ew", padx=5)
        self.btn_save = self._button(action_row, "⇩ Sonucu Kaydet", self._save_result, compact=True, pack=False)
        self.btn_save.grid(row=0, column=3, sticky="ew", padx=5)
        self.btn_steps = self._button(action_row, "▤ Ara Adımlar", self._save_steps, compact=True, pack=False)
        self.btn_steps.grid(row=0, column=4, sticky="ew", padx=5)
        self.btn_stop = self._button(action_row, "■ Durdur", self._stop_video, danger=True, compact=True, pack=False)
        self.btn_stop.grid(row=0, column=5, sticky="ew", padx=(5, 0))
        self.btn_stop.configure(state="disabled")

        self._rail_card(rail, "✓", "İnceleme Durumu", self.dashboard_detection,
                        "Sol ve sağ sınır durumu", GOOD, large=False)
        overview_grid = tk.Frame(rail, bg=BG)
        overview_grid.pack(fill="x", pady=(0, 7))
        for column in range(2):
            overview_grid.grid_columnconfigure(column, weight=1)
        self._rail_mini_card(overview_grid, "◉", "Görsel Destek", self.dashboard_confidence,
                             GOOD, 0, 0, padx=(0, 4))
        self._rail_mini_card(overview_grid, "◈", "Merkez Ofseti", self.metric_offset,
                             BLUE, 0, 1, padx=(4, 0))
        rail_grid = tk.Frame(rail, bg=BG)
        rail_grid.pack(fill="x", pady=(0, 7))
        for column in range(2):
            rail_grid.grid_columnconfigure(column, weight=1)
        self._rail_mini_card(rail_grid, "⌁", "Akış", self.dashboard_fps, BLUE, 0, 0, padx=(0, 4))
        self._rail_mini_card(rail_grid, "⌛", "Süre", self.dashboard_latency, GOOD, 0, 1, padx=(4, 0))
        self._rail_mini_card(rail_grid, "⌁", "Kararlılık", self.dashboard_stability, GOOD, 1, 0, padx=(0, 4), pady=(7, 0))
        self._rail_mini_card(rail_grid, "◈", "Yol Geometrisi", self.dashboard_curve, WARM, 1, 1, padx=(4, 0), pady=(7, 0))
        alert = tk.Frame(rail, bg=PANEL, padx=14, pady=13, highlightbackground=BORDER, highlightthickness=1)
        alert.pack(fill="x", pady=(0, 10))
        tk.Label(alert, text="⚠  Durum Notları", bg=PANEL, fg=WARM,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(alert, textvariable=self.dashboard_alert, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), justify="left", wraplength=255).pack(anchor="w", pady=(8, 0))
        trend = tk.Frame(rail, bg=PANEL, padx=11, pady=9, highlightbackground=BORDER, highlightthickness=1)
        trend.pack(fill="x", pady=(0, 10))
        tk.Label(trend, text="⌁  Merkez Ofseti · Son Ölçümler", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.offset_chart = tk.Canvas(trend, bg=PANEL_DARK, height=66, highlightthickness=0, bd=0)
        self.offset_chart.pack(fill="x", pady=(6, 0))
        self.offset_chart.bind("<Configure>", lambda *_: self._draw_offset_chart())
        tk.Label(rail, textvariable=self.algorithm_method, bg=BG, fg=GOOD,
                 font=("Segoe UI", 8, "bold"), justify="left", wraplength=275).pack(fill="x", pady=(5, 0))

        footer = tk.Frame(app, bg=BG)
        footer.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(footer, textvariable=self.status, bg=BG, fg=GOOD, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(footer, text="Eğitim ve araştırma amaçlıdır; gerçek araç kontrolünde kullanılmaz.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="right")

    def _nav_button(self, parent, label, command, active=False):
        """Sol gezinti çubuğunda seçili durumu korunan düğme oluştur."""
        normal = "#1E4268" if active else SIDEBAR
        button = tk.Button(parent, text=label, command=lambda: None, anchor="w",
                           bg=normal, fg=TEXT, activebackground=PANEL_HOVER,
                           activeforeground=TEXT, bd=0, relief="flat", padx=16, pady=11,
                           font=("Segoe UI", 10, "bold" if active else "normal"), cursor="hand2")

        def activate():
            for item in self._nav_buttons:
                item.configure(bg=SIDEBAR, font=("Segoe UI", 10, "normal"))
            button.configure(bg="#1E4268", font=("Segoe UI", 10, "bold"))
            command()

        button.configure(command=activate)
        button.pack(fill="x", pady=2)
        self._nav_buttons.append(button)
        return button

    def _top_badge(self, parent, text, color):
        badge = tk.Frame(parent, bg=PANEL, padx=12, pady=7, highlightbackground=BORDER, highlightthickness=1)
        badge.pack(side="left", padx=(7, 0))
        options = {"bg": PANEL, "fg": color, "font": ("Segoe UI", 9, "bold")}
        if isinstance(text, tk.Variable):
            options["textvariable"] = text
        else:
            options["text"] = text
        tk.Label(badge, **options).pack()

    def _scene_chip(self, parent, variable, color):
        chip = tk.Frame(parent, bg=PANEL_HOVER, padx=10, pady=5)
        chip.pack(side="left", padx=(0, 7))
        tk.Label(chip, textvariable=variable, bg=PANEL_HOVER, fg=color,
                 font=("Segoe UI", 9, "bold")).pack()

    def _rail_card(self, parent, icon, title, value, subtitle, accent, large=False):
        card = tk.Frame(parent, bg=PANEL, padx=12, pady=9, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 7))
        header = tk.Frame(card, bg=PANEL)
        header.pack(fill="x")
        tk.Label(header, text=icon, bg=PANEL, fg=accent, font=("Segoe UI Symbol", 16, "bold")).pack(side="left")
        tk.Label(header, text=title, bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(9, 0))
        tk.Label(card, textvariable=value, bg=PANEL, fg=accent if large else TEXT,
                 font=("Segoe UI", 17 if large else 13, "bold"), anchor="w").pack(fill="x", pady=(4, 1))
        tk.Label(card, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(fill="x")
        return card

    def _rail_mini_card(self, parent, icon, title, value, accent, row, column, padx=0, pady=0):
        card = tk.Frame(parent, bg=PANEL, padx=9, pady=7, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=row, column=column, sticky="nsew", padx=padx, pady=pady)
        tk.Label(card, text=f"{icon}  {title}", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x")
        tk.Label(card, textvariable=value, bg=PANEL, fg=accent,
                 font=("Segoe UI", 13, "bold"), anchor="w").pack(fill="x", pady=(3, 0))
        return card

    def _stage_tab(self, parent, label, stage):
        button = tk.Button(parent, text=label, command=lambda: self.selected_stage.set(stage),
                           bg=PANEL, fg=MUTED, activebackground=PANEL_HOVER, activeforeground=TEXT,
                           bd=0, padx=11, pady=6, font=("Segoe UI", 8, "bold"), cursor="hand2")
        button.pack(side="left", padx=(0, 5))
        self._stage_tab_buttons[stage] = button
        return button

    def _stage_changed(self, *_):
        display_label = STAGE_LABELS.get(self.selected_stage.get(), self.selected_stage.get())
        if self.stage_label.get() != display_label:
            self.stage_label.set(display_label)
        self._refresh_stage()
        for stage, button in getattr(self, "_stage_tab_buttons", {}).items():
            active = stage == self.selected_stage.get()
            button.configure(bg="#244A68" if active else PANEL, fg=TEXT if active else MUTED)

    def _refresh_video_training_status(self):
        try:
            stats = video_training_stats(APP_ROOT / "data" / "gercek")
        except OSError:
            stats = {"verified_video_frames": 0, "source_videos": 0,
                     "recommended_frames": 10, "recommended_videos": 3, "ready": False}
        frames = stats["verified_video_frames"]
        videos = stats["source_videos"]
        self.video_training_var.set(
            f"{frames}/{stats['recommended_frames']} doğrulanmış kare · {videos}/{stats['recommended_videos']} video"
        )
        self.video_training_detail.set(
            "Video eğitim seti hazır" if stats["ready"]
            else "Kare seçin, düz/kesik çizgileri düzeltin ve eğitime ekleyin."
        )
        return stats

    def _show_video_training_info(self):
        stats = self._refresh_video_training_status()
        messagebox.showinfo(
            "Video eğitim kampanyası",
            f"Doğrulanmış video karesi: {stats['verified_video_frames']}\n"
            f"Farklı kaynak video: {stats['source_videos']}\n\n"
            "Ham video kareleri otomatik olarak eğitim etiketi sayılmaz. Her videodan "
            "farklı bir kare seçin; hatalıysa düz/kesik sınırları ve aradaki alanı çizin. "
            "Kaydedilen kareler, aynı video grubu korunarak yerel eğitime katılır.",
        )

    def _tick_dashboard(self):
        elapsed = max(0, int(time.monotonic() - self._started_at))
        self.dashboard_runtime.set(f"{elapsed // 3600:02d}:{(elapsed // 60) % 60:02d}:{elapsed % 60:02d}")
        self.after(1000, self._tick_dashboard)

    def _append_offset(self, offset):
        if offset is None:
            return
        self._offset_history.append(float(offset))
        self._offset_history = self._offset_history[-60:]
        if len(self._offset_history) < 3:
            self.dashboard_stability.set("Ölçülüyor")
        else:
            deviation = float(np.std(self._offset_history[-20:]))
            score = max(0.0, 100.0 - min(100.0, deviation * 2.5))
            self.dashboard_stability.set(f"%{score:.1f}")
        self._draw_offset_chart()

    def _draw_offset_chart(self):
        if not hasattr(self, "offset_chart"):
            return
        canvas = self.offset_chart
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        if not self._offset_history:
            canvas.create_text(width / 2, height / 2, text="Merkez sapması ölçümü bekleniyor",
                               fill=MUTED, font=("Segoe UI", 8))
            return
        values = self._offset_history
        scale = max(20.0, max(abs(value) for value in values) * 1.15)
        mid = height / 2
        canvas.create_line(0, mid, width, mid, fill=BORDER, dash=(3, 3))
        points = []
        for index, value in enumerate(values):
            x = 5 + (width - 10) * index / max(1, len(values) - 1)
            y = mid - (value / scale) * (height * .39)
            points.extend((x, y))
        if len(points) >= 4:
            canvas.create_line(*points, fill=GOOD, width=2, smooth=True)
        canvas.create_text(width - 6, 7, text=f"{values[-1]:+.1f} px", fill=GOOD,
                           anchor="ne", font=("Segoe UI", 8, "bold"))

    def _record_status(self, *_):
        """Alt durum ile sağ raydaki uyarı ve kısa olay günlüğünü eşleştir."""
        message = self.status.get().strip()
        if not message:
            return
        self.dashboard_alert.set(message)
        if not hasattr(self, "activity_log"):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"●  [{timestamp}]  {message}"
        if self._activity_lines and self._activity_lines[-1].endswith(message):
            return
        self._activity_lines.append(line)
        self._activity_lines = self._activity_lines[-4:]
        self.activity_log.configure(state="normal")
        self.activity_log.delete("1.0", "end")
        self.activity_log.insert("end", "\n".join(self._activity_lines))
        self.activity_log.configure(state="disabled")

    def _toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill="x", pady=(8, 0))
        else:
            self.advanced_frame.pack_forget()
        self.advanced_btn.configure(text="⚙ Ayarları Gizle" if self.advanced_visible else "⚙ Ayarlar")

    def _analyze_loaded_scene(self, frame):
        try:
            p=analyze_scene(frame)
            self.scene_profile=p
            self.scene_lighting.set("☾  "+p.short_lighting if p.lighting=='gece' else "☀  "+p.short_lighting)
            self.scene_geometry.set("↝  "+p.short_geometry)
            self.scene_brightness.set(f"Işık endeksi {p.brightness:.0f}/255  ·  Viraj işareti: {p.curve_score:.1f}")
            self.dashboard_curve.set(f"{p.short_geometry} · {p.curve_score:.1f}")
        except Exception:
            self.scene_profile=None
            self.scene_lighting.set("IŞIK: BELİRSİZ")
            self.scene_geometry.set("YOL: BELİRSİZ")
            self.scene_brightness.set("Yeterli görsel veri bulunamadı")
            self.dashboard_curve.set("Belirsiz")

    def _subhead(self, parent, text):
        background = parent.cget("bg")
        tk.Label(parent, text=text, bg=background, fg=BLUE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(9, 8))

    def _button(self, parent, label, command, primary=False, danger=False, compact=False, pack=True):
        normal = DANGER if danger else (BLUE if primary else PANEL)
        hover = "#C75A60" if danger else ("#5B9FD2" if primary else PANEL_HOVER)
        foreground = TEXT if danger else ("#031321" if primary else TEXT)
        size, padding = (9, 7) if compact else (10, 10)
        btn = tk.Button(parent, text=label, command=command,
                        bg=normal, fg=foreground, activebackground=hover,
                        activeforeground=foreground, bd=0, highlightthickness=0, relief="flat",
                        font=("Segoe UI", size, "bold"), padx=11, pady=padding, cursor="hand2")
        btn.bind("<Enter>", lambda _event: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda _event: btn.configure(bg=normal))
        if pack:
            btn.pack(fill="x", pady=(0, 5))
        return btn

    def _slider(self, parent, label, key, minimum, maximum):
        background = parent.cget("bg")
        row = tk.Frame(parent, bg=background)
        row.pack(fill="x")
        value = self.settings_vars[key]
        tk.Label(row, text=label, fg=MUTED, bg=background,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(row, textvariable=value, fg=TEXT, bg=background,
                 font=("Segoe UI", 9, "bold")).pack(side="right")
        tk.Scale(parent, from_=minimum, to=maximum, variable=value, orient="horizontal",
                 bg=background, fg=TEXT, troughcolor=PANEL_DARK, highlightthickness=0,
                 activebackground=BLUE, bd=0, showvalue=False, sliderlength=14,
                 sliderrelief="flat").pack(fill="x", pady=(0, 1))

    def _settings(self) -> LaneSettings:
        v = self.settings_vars
        settings = LaneSettings(canny_low=v["canny_low"].get(),
                                canny_high=v["canny_high"].get(),
                                roi_top=v["roi_top"].get() / 100.0,
                                hough_threshold=v["hough_threshold"].get(),
                                min_line_length=v["min_line_length"].get(),
                                max_line_gap=v["max_line_gap"].get(),
                                smoothing=v["smoothing"].get() / 100.0)
        settings.validate()
        return settings

    def _load_example_if_exists(self, force=False):
        example = APP_ROOT / "ornekler" / "ornek_yol_duz.png"
        if example.is_file() and not self.is_processing and (force or self.current_file is None):
            self._open_image(example)

    def _select_image(self):
        if self.is_processing:
            messagebox.showwarning("İşlem sürüyor", "Önce mevcut video işlemini durdurun.")
            return
        if self.annotation_in_progress:
            messagebox.showwarning("Etiketleme sürüyor", "Önce açık etiketleme penceresinde kaydedin veya çıkın.")
            return
        self._close_live_video()
        name = filedialog.askopenfilename(title="Yol görüntüsü seç", filetypes=[
            ("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tüm dosyalar", "*.*")])
        if name:
            self._open_image(Path(name))

    def _select_video(self):
        if self.is_processing:
            messagebox.showwarning("İşlem sürüyor", "Önce mevcut video işlemini durdurun.")
            return
        if self.annotation_in_progress:
            messagebox.showwarning("Etiketleme sürüyor", "Önce açık etiketleme penceresinde kaydedin veya çıkın.")
            return
        self._close_live_video()
        name = filedialog.askopenfilename(title="Yol videosu seç", filetypes=[
            ("Videolar", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Tüm dosyalar", "*.*")])
        if not name:
            return
        self.current_file = Path(name)
        self.current_image = None
        self.current_detection = None
        self.source_name.set(self.current_file.name)
        cap = cv2.VideoCapture(str(self.current_file))
        try:
            total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if cap.isOpened() else 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
            ok, frame = cap.read() if cap.isOpened() else (False, None)
        finally:
            cap.release()
        if ok and frame is not None:
            self.current_image = frame
            self.current_display = frame
            self._analyze_loaded_scene(frame)
            self.selected_stage.set("01 Orijinal")
            self._render_preview()
        else:
            messagebox.showerror("Hata", "Video önizlemesi açılamadı.")
            self.current_file = None
            return
        self.btn_live.configure(state="normal", text="▶ CANLI VİDEOYU BAŞLAT")
        self.btn_stop.configure(state="disabled")
        self._configure_video_seek(total, 0, enabled=True)
        self.status.set("Video seçildi. Canlı kaplamayı başlatın veya videoyu dosyaya işleyin.")
        self.progress_percent.set(0)
        self.progress_text.set("Canlı video başlatılmayı bekliyor")
        self.dashboard_fps.set(f"{fps:.1f} FPS · Hazır" if fps > 0 else "Video hazır")
        self._clear_metrics()

    def _open_image(self, path: Path):
        self._close_live_video()
        try:
            self.current_image = read_image(path)
        except Exception as exc:
            messagebox.showerror("Dosya hatası", str(exc))
            return
        self.current_file = path
        self._configure_video_seek(0, 0, enabled=False)
        self.btn_live.configure(state="disabled", text="▶ CANLI VİDEOYU BAŞLAT")
        self.btn_stop.configure(state="disabled")
        self.current_detection = None
        self.source_name.set(path.name)
        self.current_display = self.current_image
        self._analyze_loaded_scene(self.current_image)
        self.selected_stage.set("01 Orijinal")
        self.progress_percent.set(0)
        self.progress_text.set("Görüntü işlenmeyi bekliyor")
        self.dashboard_fps.set("Tek kare")
        self.status.set(f"Görüntü açıldı: {path.name}. Tespit düğmesine basın.")
        self._clear_metrics()
        self._render_preview()
        saved_review=find_saved_review(self.current_image,APP_ROOT / "data" / "gercek")
        if saved_review is not None:
            self._show_saved_review(saved_review["line_mask"],saved_review.get("area_mask"),recalled=True)
            return
        if self.auto_detect_var.get():
            self.after(80, self._detect)

    def _configure_video_seek(self, total: int, index: int, enabled: bool):
        """Keep the frame slider in sync without treating programmatic moves as user seeks."""
        self.video_total_frames = max(0, int(total))
        self.video_selected_index = max(0, min(int(index), max(0, self.video_total_frames - 1)))
        self.video_seek_sync = True
        self.video_seek.configure(to=max(1, self.video_total_frames - 1), state="normal" if enabled else "disabled")
        self.video_frame_var.set(self.video_selected_index)
        self.video_seek_sync = False
        self.video_frame_text.set(
            f"Seçili kare: {self.video_selected_index + 1} / {self.video_total_frames}"
            if enabled and self.video_total_frames else "Video karesi: video açın"
        )

    def _queue_video_seek(self, value):
        """Debounce slider motion so dragging does not decode every intermediate frame."""
        if self.video_seek_sync or self.annotation_in_progress or self.is_processing:
            return
        if self.current_file is None or self.current_file.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        self.video_selected_index = max(0, min(int(float(value)), max(0, self.video_total_frames - 1)))
        self.video_frame_text.set(f"Seçili kare: {self.video_selected_index + 1} / {self.video_total_frames}")
        self._close_live_video()
        self.btn_live.configure(state="normal", text="▶ SEÇİLİ KARENDEN BAŞLAT")
        if self.video_seek_after_id is not None:
            try:
                self.after_cancel(self.video_seek_after_id)
            except tk.TclError:
                pass
        self.video_seek_after_id = self.after(140, self._show_selected_video_frame)

    def _show_selected_video_frame(self):
        self.video_seek_after_id = None
        if self.current_file is None or self.current_file.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        cap = cv2.VideoCapture(str(self.current_file))
        try:
            if not cap.isOpened():
                raise ValueError("Video açılamadı.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_selected_index)
            ok, frame = cap.read()
        except Exception as exc:
            messagebox.showerror("Kare açılamadı", str(exc))
            return
        finally:
            cap.release()
        if not ok or frame is None:
            messagebox.showerror("Kare açılamadı", "Seçilen video karesi okunamadı.")
            return
        self.current_image = frame
        self.current_detection = None
        self.current_display = frame
        self.selected_stage.set("01 Orijinal")
        self._analyze_loaded_scene(frame)
        self._clear_metrics()
        self.progress_percent.set(100 * (self.video_selected_index + 1) / self.video_total_frames if self.video_total_frames else 0)
        self.progress_text.set(f"Seçilen kare: {self.video_selected_index + 1} / {self.video_total_frames}")
        self.dashboard_fps.set("Kare seçildi")
        self.status.set("Kare seçildi. Etiketleyebilir veya bu kareden canlı takibi başlatabilirsiniz.")
        self._render_preview()

    def _clear_metrics(self):
        for v in (self.metric_left, self.metric_right, self.metric_offset, self.metric_segments):
            v.set("—")
        self.dashboard_detection.set("Analiz bekleniyor")
        self.dashboard_confidence.set("—")
        self.dashboard_lane_count.set("0")
        self.dashboard_latency.set("—")
        self.dashboard_stability.set("—")
        self._offset_history.clear()
        self._draw_offset_chart()
        self._set_feedback_ready(False, "Özgün fotoğrafın tespitinden sonra sonucu doğru veya hatalı olarak işaretleyebilirsiniz.")

    def _set_feedback_ready(self, ready, message=None):
        state = "normal" if ready else "disabled"
        self.btn_feedback_ok.configure(state=state)
        self.btn_feedback_fix.configure(state=state)
        if message is not None:
            self.feedback_status.set(message)

    def _show_saved_review(self,line_mask,area_mask,recalled:bool=False):
        """Display a verified correction and prevent an old model from overwriting it."""
        preview=render_reviewed_annotation(self.current_image,line_mask,area_mask)
        review=Detection(preview,{"01 Orijinal":self.current_image.copy(),
                                  "11 Kullanıcı düzeltmesi":preview},None,None,None,1.,0)
        review.method="Kullanıcı doğrulaması · kaydedildi"
        self.current_detection=review
        self._set_stage_options(review.stages)
        self.selected_stage.set("11 Kullanıcı düzeltmesi")
        self.current_display=preview
        self._render_preview()
        left_type="DÜZ" if np.any(line_mask==1) else "KESİK"
        right_type="DÜZ" if np.any(line_mask==3) else "KESİK"
        self.metric_left.set(left_type+" • DOĞRULANDI")
        self.metric_right.set(right_type+" • DOĞRULANDI")
        self.metric_offset.set("ETİKETLENDİ")
        self.metric_segments.set("MANUEL")
        self.dashboard_detection.set("Kullanıcı onayıyla düzeltildi")
        self.dashboard_confidence.set("Onaylı")
        self.dashboard_lane_count.set("2")
        self.algorithm_method.set("Çalışma profili: kullanıcı doğrulaması")
        count=style_sample_count(APP_ROOT / "data" / "gercek")
        message=(f"Bu fotoğraf için kaydedilmiş kullanıcı düzeltmesi gösteriliyor ({count} etiketli fotoğraf)."
                 if recalled else f"Düzeltmeniz doğru kabul edildi ve eğitime eklendi ({count} etiketli fotoğraf).")
        self._set_feedback_ready(False,message)
        self.status.set("Kullanıcı-onaylı çizgiler ve alan gösteriliyor; eski model yeniden çalıştırılmadı.")

    def _render_preview(self):
        if self.current_display is None:
            return
        try:
            h, w = self.current_display.shape[:2]
            pw = max(100, self.image_panel.winfo_width() - 6)
            ph = max(100, self.image_panel.winfo_height() - 6)
            size = min(pw / w, ph / h)
            tw, th = max(1, int(w * size)), max(1, int(h * size))
            if self.current_display.ndim == 2:
                img = Image.fromarray(self.current_display)
            else:
                img = Image.fromarray(cv2.cvtColor(self.current_display, cv2.COLOR_BGR2RGB))
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.preview_photo, text="")
        except tk.TclError:
            return

    def _refresh_stage(self):
        if self.current_detection is not None and not self.is_processing:
            self.current_display = self.current_detection.stages.get(self.selected_stage.get(),
                                                                      self.current_detection.frame)
        elif self.current_image is not None and not self.is_processing:
            self.current_display = self.current_image
        self._render_preview()

    def _stage_label_changed(self, *_):
        """Aşama seçicisinin kullanıcı dostu adını dahili işlem anahtarına bağla."""
        selected = self.stage_label.get()
        for stage, label in STAGE_LABELS.items():
            if label == selected and stage != self.selected_stage.get():
                self.selected_stage.set(stage)
                return

    def _set_stage_options(self, stages):
        """Yalnızca mevcut aşamaları, teknik adları göstermeden listeler."""
        labels = [STAGE_LABELS.get(stage, stage) for stage in stages]
        self.stage_combo.configure(values=labels)

    def _display_detection_method(self, method):
        """Teknik motor adını ana ekranda sade bir çalışma profili olarak göster."""
        value = str(method).casefold()
        if any(name in value for name in ("yolopv2", "ufld", "u-net", "model")):
            return "Çalışma profili: gelişmiş çizgi analizi"
        if any(name in value for name in ("canny", "hough", "klasik")):
            return "Çalışma profili: çizgi ve yol geometrisi"
        return "Çalışma profili: otomatik yol incelemesi"

    def _set_metrics(self, detection, elapsed_ms: float | None = None):
        marking_types=getattr(detection,"marking_types",(None,None))
        left_type,right_type=marking_types
        self.metric_left.set((left_type.upper()+" • BULUNDU") if detection.left and left_type else
                             ("BULUNDU" if detection.left else "BULUNAMADI"))
        self.metric_right.set((right_type.upper()+" • BULUNDU") if detection.right and right_type else
                              ("BULUNDU" if detection.right else "BULUNAMADI"))
        self.metric_offset.set(f"{detection.offset_px:+.1f} px" if detection.offset_px is not None else "ÖLÇÜLEMEDİ")
        self.metric_segments.set(str(detection.segment_count) if detection.segment_count else "—")
        if detection.both_found:
            self.dashboard_detection.set("Şeritler tespit edildi")
            self.dashboard_lane_count.set("2")
        elif detection.left is not None or detection.right is not None:
            self.dashboard_detection.set("Tek şerit tespit edildi")
            self.dashboard_lane_count.set("1")
        else:
            self.dashboard_detection.set("Şerit bulunamadı")
            self.dashboard_lane_count.set("0")
        self.dashboard_confidence.set(f"%{max(0.0, min(1.0, detection.confidence)) * 100:.0f}")
        if elapsed_ms is not None:
            self.dashboard_latency.set(f"~{elapsed_ms:.0f} ms")
        self._append_offset(detection.offset_px)
        self.algorithm_method.set(self._display_detection_method(detection.method))
        profile=getattr(detection,"scene_profile",None)
        if profile is not None:
            self.scene_profile=profile
            self.scene_lighting.set(("☾  " if profile.lighting=='gece' else "☀  ")+profile.short_lighting)
            self.scene_geometry.set("↝  "+profile.short_geometry)
            self.scene_brightness.set(f"Işık endeksi {profile.brightness:.0f}/255  ·  Viraj işareti: {profile.curve_score:.1f}")
            self.dashboard_curve.set(f"{profile.short_geometry} · {profile.curve_score:.1f}")

    def _detect(self):
        if self.is_processing:
            return
        if self.current_file is None:
            messagebox.showinfo("Dosya seçin", "Önce bir yol görüntüsü veya video seçmelisiniz.")
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Parametre hatası", str(exc))
            return
        if self.current_file.suffix.lower() in VIDEO_EXTENSIONS:
            self._run_video(settings)
        else:
            try:
                started = time.perf_counter()
                detection = InferenceEngine(settings, mode=self._mode()).detect(self.current_image)
            except Exception as exc:
                messagebox.showerror("İşlem hatası", str(exc))
                return
            self.current_detection = detection
            self._set_stage_options(detection.stages)
            self.selected_stage.set("08 Sonuc")
            self.current_display = detection.frame
            self._render_preview()
            self._set_metrics(detection, (time.perf_counter() - started) * 1000)
            self._set_feedback_ready(True, "Sonucu kontrol edin: doğruysa eğitime ekleyin, hatalıysa doğru çizgileri çizin.")
            self.progress_percent.set(100)
            self.progress_text.set("Görüntü analizi tamamlandı")
            self.status.set("Tespit tamamlandı. Çıktıyı ve ara adımları kaydedebilirsiniz.")

    def _mode(self):
        return self.mode_keys.get(self.mode_var.get(), "auto")

    def _start_live_video(self):
        """Play one video through the detector and refresh the preview per frame."""
        if self.is_processing:
            messagebox.showwarning("Video işleniyor", "Önce dosyaya işlenen videonun durmasını bekleyin.")
            return
        if self.current_file is None or self.current_file.suffix.lower() not in VIDEO_EXTENSIONS:
            messagebox.showinfo("Video seçin", "Önce bir yol videosu açın.")
            return
        if self.annotation_in_progress:
            messagebox.showinfo("Etiketleme sürüyor", "Önce açık etiketleme penceresinde kaydedin veya çıkın.")
            return
        if self.live_session is None:
            try:
                self.live_session = LiveVideoSession(self.current_file, self._settings(), self._mode(),
                                                     start_frame=self.video_selected_index,
                                                     saved_anchors=load_video_anchors(
                                                         APP_ROOT / "data" / "gercek", self.current_file))
                if self.pending_manual_anchor is not None:
                    anchor_frame, anchor_mask = self.pending_manual_anchor
                    self.live_session.set_manual_anchor(anchor_frame, anchor_mask)
                    self.pending_manual_anchor = None
            except (OSError, ValueError) as exc:
                messagebox.showerror("Canlı video açılamadı", str(exc))
                self.live_session = None
                return
            self.live_frame_index = 0
        self.live_playing = True
        self.btn_live.configure(state="disabled", text="▶ CANLI VİDEO ÇALIŞIYOR")
        self.btn_stop.configure(state="normal")
        self.btn_detect.configure(state="disabled")
        self.video_seek.configure(state="disabled")
        # Video seçilirken ilk karenin ham görünümü gösterilir. Canlı
        # takipte ise kullanıcı her zaman kaplamalı sonucu görmelidir.
        self.selected_stage.set("08 Sonuc")
        self._set_feedback_ready(False, "Canlı şerit takibi çalışıyor. Hatalı bir karede videoyu durdurun.")
        self.dashboard_fps.set("Takip aktif")
        self.status.set("Canlı video başladı; şerit kaplaması her karede güncelleniyor.")
        self._next_live_frame()

    def _next_live_frame(self):
        """Advance the Tk event loop only after one frame has been fully analysed."""
        self.live_after_id = None
        if not self.live_playing or self.live_session is None:
            return
        try:
            started = time.perf_counter()
            packet = self.live_session.read_next()
        except Exception as exc:
            self._finish_live_video(error=str(exc))
            return
        if packet is None:
            self._finish_live_video(completed=True)
            return
        self.live_frame_index = packet.index
        self.video_selected_index = max(0, packet.index - 1)
        self.video_seek_sync = True
        self.video_frame_var.set(self.video_selected_index)
        self.video_seek_sync = False
        self.video_frame_text.set(f"Canlı kare: {packet.index} / {packet.total if packet.total else '?'}")
        self.current_image = packet.original
        self.current_detection = packet.detection
        self._set_stage_options(packet.detection.stages)
        if self.selected_stage.get() not in packet.detection.stages:
            self.selected_stage.set("08 Sonuc")
        self.current_display = packet.detection.stages.get(self.selected_stage.get(), packet.detection.frame)
        self._render_preview()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._set_metrics(packet.detection, elapsed_ms)
        stability_note = packet.stability_note or ""
        if packet.anchored:
            source = "Doğrulanmış çizim izleniyor."
        elif stability_note == "ani sıçrama engellendi":
            source = "Ani çizgi ve alan sapması engellendi; son güvenilir geometri korunuyor."
        elif stability_note == "ölçüm bekleniyor":
            source = "Geçici ölçüm kaybında son güvenilir şerit çifti korunuyor."
        elif stability_note == "yeni geometri onaylandı":
            source = "Yeni yol geometrisi ardışık ölçümlerle doğrulandı."
        elif packet.stabilized:
            source = "Şerit geometrisi kareler arasında yumuşatılarak takip ediliyor."
        else:
            source = "Hatalıysa videoyu durdurup doğru çizgileri işaretleyin."
        self._set_feedback_ready(True, f"Canlı kare {packet.index}. {source}")
        actual_fps = min(packet.fps, 1000.0 / max(elapsed_ms, 1.0))
        self.dashboard_fps.set(f"{actual_fps:.1f} FPS")
        self.progress_percent.set(min(100, 100 * packet.index / packet.total) if packet.total else 0)
        self.progress_text.set(f"Canlı kare: {packet.index} / {packet.total if packet.total else '?'}")
        delay = max(1, int(1000 / packet.fps - elapsed_ms))
        self.live_after_id = self.after(delay, self._next_live_frame)

    def _pause_live_video(self, message: str | None = None):
        """Freeze the displayed original frame so it can safely be reviewed."""
        if not self.live_playing:
            return
        self.live_playing = False
        if self.live_after_id is not None:
            try:
                self.after_cancel(self.live_after_id)
            except tk.TclError:
                pass
            self.live_after_id = None
        self.btn_live.configure(state="normal", text="▶ CANLI VİDEOYA DEVAM ET")
        self.btn_stop.configure(state="disabled")
        self.btn_detect.configure(state="normal")
        self.video_seek.configure(state="normal")
        self.dashboard_fps.set("Duraklatıldı")
        if self.current_detection is not None:
            self._set_feedback_ready(True, f"Kare {self.live_frame_index} duraklatıldı. Doğruysa kaydedin, hatalıysa çizin.")
        self.status.set(message or f"Canlı video {self.live_frame_index}. karede duraklatıldı.")

    def _close_live_video(self):
        """Release a live capture without touching a separately saved batch output."""
        self.live_playing = False
        if self.live_after_id is not None:
            try:
                self.after_cancel(self.live_after_id)
            except tk.TclError:
                pass
            self.live_after_id = None
        if self.live_session is not None:
            self.live_session.close()
            self.live_session = None
        self.pending_manual_anchor = None
        if self.video_seek_after_id is not None:
            try:
                self.after_cancel(self.video_seek_after_id)
            except tk.TclError:
                pass
            self.video_seek_after_id = None
        if hasattr(self, "btn_live"):
            self.btn_live.configure(state="disabled", text="▶ CANLI VİDEOYU BAŞLAT")
            self.btn_stop.configure(state="disabled")
            if not self.is_processing:
                self.btn_detect.configure(state="normal")

    def _finish_live_video(self, completed: bool = False, error: str | None = None):
        self._close_live_video()
        self.btn_live.configure(state="normal", text="▶ CANLI VİDEOYU BAŞTAN BAŞLAT")
        if self.video_total_frames:
            self.video_seek.configure(state="normal")
        if error:
            self.dashboard_fps.set("Akış hatası")
            self.status.set("Canlı video sırasında hata oluştu.")
            messagebox.showerror("Canlı video hatası", error)
        elif completed:
            self.dashboard_fps.set("Tamamlandı")
            self.progress_percent.set(100)
            self._configure_video_seek(self.video_total_frames, 0, enabled=True)
            self.status.set("Canlı video tamamlandı. Baştan oynatabilir veya son kareyi değerlendirebilirsiniz.")
        else:
            self.dashboard_fps.set("Hazır")
            self.status.set("Canlı video kapatıldı.")

    def _run_video(self, settings):
        self._close_live_video()
        name = filedialog.asksaveasfilename(title="Video çıktısının konumu", defaultextension=".mp4",
                                            initialfile=self.current_file.stem + "_sonuc.mp4",
                                            filetypes=[("MP4 video", "*.mp4")])
        if not name:
            return
        dest = Path(name)
        if dest.resolve() == self.current_file.resolve():
            messagebox.showerror("Dosya hatası", "Giriş videosunun üzerine yazılamaz.")
            return
        if dest.is_file() and not messagebox.askyesno("Üzerine yaz", "Bu çıktı zaten var. Üzerine yazılsın mı?"):
            return
        self.output_video = dest
        chosen_mode = self._mode()
        self.stop_event.clear()
        self.is_processing = True
        self.btn_detect.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.video_seek.configure(state="disabled")
        self.progress_percent.set(0)
        self.progress_text.set("Video işleniyor...")
        self.dashboard_fps.set("Dosyaya işleniyor")
        self.status.set("Video kare kare işleniyor. İşlem sırasında arayüz kullanılabilir.")

        def worker():
            try:
                def on_progress(current, total, detected):
                    # Kuyruğun dolması işleyiciyi durdurmaz; yalnızca eski kare atlanır.
                    try:
                        self.messages.put_nowait(("progress", current, total, detected))
                    except queue.Full:
                        pass
                summary = process_video(self.current_file, dest, settings, on_progress,
                                        self.stop_event.is_set, mode=chosen_mode)
                self.messages.put(("complete", summary))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _process_messages(self):
        try:
            while True:
                item = self.messages.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, current, total, detection = item
                    self.current_display = detection.frame
                    self._render_preview()
                    self._set_metrics(detection)
                    self.progress_percent.set(min(100, 100 * current / total) if total else 0)
                    self.progress_text.set(f"İşlenen kare: {current} / {total if total else '?'}")
                elif kind == "complete":
                    summary = item[1]
                    self._finish_video()
                    self.progress_text.set(f"{summary.frames} kare işlendi • İki şerit oranı: "
                                           f"{summary.to_dict()['iki_serit_tespit_orani_yuzde']}%")
                    self.status.set("İşlem durduruldu; kısmi çıktı kaydedildi." if summary.cancelled else
                                    f"Video tamamlandı: {Path(summary.output_path).name} (CSV + JSON rapor hazır)")
                    if not summary.cancelled:
                        messagebox.showinfo("Tamamlandı", "Video ve istatistik dosyaları kaydedildi:\n" +
                                            summary.output_path)
                elif kind == "error":
                    self._finish_video()
                    self.status.set("Video işleme sırasında hata oluştu.")
                    messagebox.showerror("İşlem hatası", item[1])
        except queue.Empty:
            pass
        self.after(100, self._process_messages)

    def _finish_video(self):
        self.is_processing = False
        self.btn_detect.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if self.current_file is not None and self.current_file.suffix.lower() in VIDEO_EXTENSIONS:
            self.btn_live.configure(state="normal", text="▶ CANLI VİDEOYU BAŞLAT")
            self.video_seek.configure(state="normal")

    def _stop_video(self):
        if self.is_processing:
            self.stop_event.set()
            self.status.set("Durdurma isteği gönderildi, mevcut kare tamamlanıyor.")
        elif self.live_playing:
            self._pause_live_video()

    def _save_result(self):
        if self.current_detection is None:
            if self.output_video is not None and self.output_video.is_file():
                messagebox.showinfo("Çıktı", f"Video zaten otomatik kaydedildi:\n{self.output_video}")
            else:
                messagebox.showinfo("Sonuç yok", "Önce bir görüntü üzerinde tespit yapın.")
            return
        name = filedialog.asksaveasfilename(title="Sonucu kaydet", defaultextension=".png",
                                            initialfile="serit_tespiti_sonuc.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if name:
            try:
                write_image(name, self.current_detection.frame)
                self.status.set("Görüntü kaydedildi: " + name)
            except Exception as exc:
                messagebox.showerror("Kaydetme hatası", str(exc))

    def _save_steps(self):
        if self.current_detection is None:
            messagebox.showinfo("Sonuç yok", "Ara adımları kaydetmek için görüntü tespiti yapın.")
            return
        folder = filedialog.askdirectory(title="Ara adımları kaydedilecek klasör")
        if folder:
            try:
                save_stages(self.current_detection.stages, folder)
                self.status.set(f"{len(self.current_detection.stages)} işlem aşaması kaydedildi: " + folder)
            except Exception as exc:
                messagebox.showerror("Kaydetme hatası", str(exc))

    def _confirm_detection(self):
        """Save only a user-validated, complete and typed result as training data."""
        if self.current_image is None or self.current_detection is None or self.current_file is None:
            messagebox.showinfo("Tespit yok", "Önce bir fotoğraf veya video karesinde şerit tespiti yapın.")
            return
        self._pause_live_video("Canlı video doğrulama için duraklatıldı.")
        try:
            sample = save_confirmed_detection(
                self.current_image, self.current_detection, APP_ROOT / "data" / "gercek", self.current_file.name)
        except ValueError as exc:
            messagebox.showinfo("Elle düzeltme gerekli", str(exc))
            self._review_incorrect()
            return
        if self.current_file.suffix.lower() in VIDEO_EXTENSIONS and self.live_session is not None:
            # Kullanıcı mevcut tespiti doğru onayladıysa, aynı sonuç sonraki
            # karelerde de takip ankrajı olmalıdır; kare kare yeniden seçim
            # yapmak görüntünün titremesine neden olur.
            saved_mask = cv2.imdecode(np.fromfile(str(sample["mask"]), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if saved_mask is not None and saved_mask.ndim == 2:
                try:
                    self.live_session.set_manual_anchor(self.current_image, saved_mask)
                except ValueError:
                    pass
        count = style_sample_count(APP_ROOT / "data" / "gercek")
        anchor_note = (" Video karesi sonraki görüntüler için takip ankrajı oldu."
                       if self.current_file.suffix.lower() in VIDEO_EXTENSIONS else "")
        self._set_feedback_ready(False, f"Onaylı örnek kaydedildi ({count} etiketli fotoğraf). "
                                  "Hatalı örneklerde 'doğrusunu çiz' seçeneğini kullanın." + anchor_note)
        self.status.set(f"Onaylı eğitim örneği yerelde kaydedildi: {sample['image'].name}")

    def _review_incorrect(self):
        self._pause_live_video("Canlı video düzeltme için duraklatıldı.")
        self._launch_labeler(correction=True)

    def _train_feedback_model(self):
        """İşaretlenmiş kullanıcı örneklerinden yerel doğrulama profilini güncelle."""
        count=style_sample_count(APP_ROOT / "data" / "gercek")
        video_stats = self._refresh_video_training_status()
        if count<24:
            messagebox.showinfo("Daha fazla etiket gerekli",
                                f"Doğrulama profilini güncellemek için en az 24 tam etiketli fotoğraf gerekir. Şu anda: {count}.")
            return
        if not messagebox.askyesno(
            "Doğrulama verisini işle",
            f"{count} yerel etiket ve {video_stats['verified_video_frames']} doğrulanmış video karesiyle "
            "düz/kesik çizgi doğrulama profili güncellenecek. Ham video kareleri otomatik eklenmez; yalnızca "
            "onaylanmış kareler kullanılır. Mevcut çalışma profili korunur. Devam edilsin mi?"
        ):
            return
        try:
            subprocess.Popen([
                sys.executable, str(APP_ROOT / "egit_yolopv2_geri_bildirim.py"),
            ], cwd=str(APP_ROOT))
        except OSError as exc:
            messagebox.showerror("Eğitim başlatılamadı", str(exc))
            return
        self.feedback_status.set("Doğrulama verisi arka planda işleniyor. Mevcut çalışma profili korunur; güncelleme tamamlandığında yeni doğrulamalar kullanılır.")
        self.status.set("Doğrulama profili güncelleniyor. Uygulamayı kullanmaya devam edebilirsiniz.")

    def _label_current_image(self):
        self._launch_labeler(correction=False)

    def _launch_labeler(self, correction: bool):
        """Fotoğrafı veya duraklatılan video karesini dört sınıfta etiketle."""
        if self.current_file is None or self.current_image is None:
            messagebox.showinfo("Görüntü seçin", "Önce etiketlemek istediğiniz yol görüntüsünü veya video karesini açın.")
            return
        if self.annotation_in_progress:
            messagebox.showinfo("Etiketleme sürüyor", "Açık etiketleme penceresinde önce kaydedin veya çıkın.")
            return
        video_frame = self.current_file.suffix.lower() in VIDEO_EXTENSIONS
        if video_frame:
            self._pause_live_video("Canlı video etiketleme için duraklatıldı.")
        if not correction and not messagebox.askyesno(
            "Eğitim etiketi",
            "Açılan pencerede 1=sol düz, 2=sol kesik, 3=sağ düz, 4=sağ kesik, 5=aradaki alan seçilir.\n"
            "Her çizgi için en az iki nokta koyun. Özgün görsel yalnızca yerel eğitim klasörüne kaydedilecek."
        ):
            return
        now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if video_frame:
            frame_dir = APP_ROOT / "data" / "gercek" / "video_kareleri"
            frame_dir.mkdir(parents=True, exist_ok=True)
            source_path = frame_dir / f"{self.current_file.stem}_kare_{self.live_frame_index:06d}_{now}.png"
            try:
                write_image(source_path, self.current_image)
            except Exception as exc:
                messagebox.showerror("Kare kaydedilemedi", str(exc))
                return
            name = source_path.stem
        else:
            source_path = self.current_file
            name = f"{self.current_file.stem}_{now}"
        try:
            process=subprocess.Popen([
                sys.executable, str(APP_ROOT / "etiketle.py"), "--image", str(source_path),
                "--name", name, "--outdir", str(APP_ROOT / "data" / "gercek"), "--style-aware",
            ], cwd=str(APP_ROOT))
        except OSError as exc:
            messagebox.showerror("Etiketleme açılamadı", str(exc))
            return
        self.annotation_in_progress = True
        if video_frame:
            self.btn_live.configure(state="disabled")
        self._set_feedback_ready(False, "Etiketleme penceresinde doğru çizgileri işaretleyip S ile kaydedin.")
        self.status.set("Etiketleme penceresi açıldı. Kaydedilen özgün görsel yalnızca yerel eğitim verisine eklenir.")
        self.after(250, lambda: self._wait_for_annotation(process, name, source_path, video_frame))

    def _wait_for_annotation(self, process, name: str, source_path: Path, video_frame: bool = False):
        """Return a saved correction to the main view after the annotator closes."""
        if process.poll() is None:
            self.after(250, lambda: self._wait_for_annotation(process, name, source_path, video_frame))
            return
        self.annotation_in_progress = False
        if video_frame and not self.is_processing:
            self.btn_live.configure(state="normal", text="▶ CANLI VİDEOYA DEVAM ET")
        root=APP_ROOT / "data" / "gercek"
        mask_path=root / "masks" / f"{name}_mask.png"
        area_path=root / "areas" / f"{name}_area.png"
        same_source = video_frame or self.current_file == source_path
        if not mask_path.is_file() or self.current_image is None or not same_source:
            if self.current_detection is not None:
                self._set_feedback_ready(True, "Etiketleme kaydedilmedi. Sonucu yeniden değerlendirebilirsiniz.")
            return
        try:
            line_mask=cv2.imdecode(np.fromfile(str(mask_path),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            area_mask=cv2.imdecode(np.fromfile(str(area_path),dtype=np.uint8),cv2.IMREAD_UNCHANGED) if area_path.is_file() else None
            if line_mask is None or line_mask.ndim!=2:
                raise ValueError("Dört sınıflı çizgi maskesi okunamadı.")
            if video_frame:
                save_video_anchor(root, self.current_file, self.live_frame_index,
                                  self.current_image, line_mask, training_name=name)
                self._refresh_video_training_status()
                if self.live_session is not None:
                    self.live_session.set_manual_anchor(self.current_image, line_mask)
                else:
                    # Video ilk karede etiketlendiyse oturum henüz açılmamış
                    # olur; ilk canlı başlatmada aynı ankraj kurulacaktır.
                    self.pending_manual_anchor = (self.current_image.copy(), line_mask.copy())
        except Exception as exc:
            messagebox.showerror("Düzeltme okunamadı", str(exc))
            return
        self._show_saved_review(line_mask,area_mask,recalled=False)

    def _on_close(self):
        if self.is_processing:
            messagebox.showwarning("Video işleniyor", "Önce 'Videoyu durdur' düğmesine basıp işlemin bitmesini bekleyin.")
            self.stop_event.set()
            return
        self._close_live_video()
        self.destroy()


def main():
    LaneApp().mainloop()
