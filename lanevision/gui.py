"""Tkinter ile Türkçe masaüstü arayüz. OpenCV worker yalnızca kuyruk üzerinden UI'yi günceller."""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .core import Detection, LaneSettings
from .engine import InferenceEngine, MODES
from .feedback import save_confirmed_detection, style_sample_count
from .processing import VIDEO_EXTENSIONS, process_video, read_image, save_stages, write_image
from .scene import analyze_scene

APP_ROOT = Path(__file__).resolve().parent.parent
BG = "#070F1D"
SIDEBAR = "#111F32"
PANEL = "#182C45"
BLUE = "#4CAEF8"
TEXT = "#F0F6FF"
MUTED = "#9EADC4"
GOOD = "#2DE2BA"
BORDER = "#253C57"


class LaneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ŞERİT ANALİZİ  |  Akıllı Yol ve Şerit Tespiti")
        self.geometry("1390x860")
        self.minsize(1050, 690)
        self.configure(bg=BG)
        self.current_file: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_detection: Detection | None = None
        self.current_display: np.ndarray | None = None
        self.output_video: Path | None = None
        self.is_processing = False
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
        self.mode_var = tk.StringVar(value=MODES["auto"])
        self.mode_keys = {label:key for key,label in MODES.items()}
        self.algorithm_method = tk.StringVar(value="Yöntem: henüz çalıştırılmadı")
        self.feedback_status = tk.StringVar(value="Tespit sonrası sonucu doğru veya hatalı olarak işaretleyebilirsiniz.")
        self.status = tk.StringVar(value="Hazır. Örnek görüntüyle başlayabilir veya kendi dosyanızı açabilirsiniz.")
        self.metric_left = tk.StringVar(value="—")
        self.metric_right = tk.StringVar(value="—")
        self.metric_offset = tk.StringVar(value="—")
        self.metric_segments = tk.StringVar(value="—")
        self.source_name = tk.StringVar(value="Dosya seçilmedi")
        self.scene_lighting = tk.StringVar(value="IŞIK: BEKLENİYOR")
        self.scene_geometry = tk.StringVar(value="YOL: BEKLENİYOR")
        self.scene_brightness = tk.StringVar(value="Görsel analizi bekleniyor")
        self.scene_profile = None
        self.ai_status_var = tk.StringVar(value="AI modeli kontrol ediliyor")
        self.auto_detect_var = tk.BooleanVar(value=True)
        self.advanced_visible = False
        self.progress_percent = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="İşlem bekleniyor")
        self._configure_styles()
        self._check_ai_ready()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._process_messages)
        self.after(200, lambda: self._load_example_if_exists(force=False))

    def _check_ai_ready(self):
        import importlib.util
        from .ai_model import DEFAULT_WEIGHTS
        if importlib.util.find_spec("torch") is None:
            self.ai_status_var.set("●  AI PASİF: PyTorch yüklü değil")
        elif not DEFAULT_WEIGHTS.is_file():
            self.ai_status_var.set("●  AI PASİF: model ağırlığı bulunamadı")
        else:
            self.ai_status_var.set("●  AI HAZIR: U-Net ağırlıkları mevcut")

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#dfeaf8", background="#dfeaf8", padding=6)
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=BLUE, borderwidth=0)

    def _build_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=26, pady=(19, 16))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand,text="◈",bg=BG,fg=GOOD,font=("Segoe UI Symbol",28,"bold")).pack(side="left",padx=(0,10))
        namebox=tk.Frame(brand,bg=BG)
        namebox.pack(side="left")
        tk.Label(namebox, text="ŞERİT ANALİZİ", bg=BG, fg=TEXT,
                 font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(namebox,text="AKILLI YOL GÖRÜNTÜSÜ ANALİZ PLATFORMU",bg=BG,fg=MUTED,
                 font=("Segoe UI",9,"bold")).pack(anchor="w")
        tk.Label(header,text="●  AI / ŞERİT DEVAMLILIĞI",bg=SIDEBAR,fg=GOOD,
                 font=("Segoe UI",10,"bold"),padx=17,pady=10).pack(side="right",pady=8)

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        # Dar ekranlarda son düğmelerin kaybolmaması için kaydırılabilir sol panel.
        sidebar_holder = tk.Frame(container, bg=SIDEBAR, width=330)
        sidebar_holder.pack(side="left", fill="y", padx=(0, 15))
        sidebar_holder.pack_propagate(False)
        scroller = tk.Canvas(sidebar_holder, bg=SIDEBAR, highlightthickness=0, borderwidth=0)
        side_scrollbar = tk.Scrollbar(sidebar_holder, orient="vertical", command=scroller.yview)
        scroller.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side="right", fill="y")
        scroller.pack(side="left", fill="both", expand=True)
        controls = tk.Frame(scroller, bg=SIDEBAR, padx=17, pady=15)
        controls_item = scroller.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _event: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda event: scroller.itemconfigure(controls_item, width=event.width))
        main = tk.Frame(container, bg=BG)
        main.pack(side="right", fill="both", expand=True)

        self._subhead(controls, "01 / DOSYA SEÇİMİ")
        self._button(controls, "GÖRÜNTÜ AÇ", self._select_image, primary=True)
        self._button(controls, "VİDEO AÇ", self._select_video)
        self._button(controls, "ÖRNEK GÖRÜNTÜYÜ AÇ", lambda: self._load_example_if_exists(force=True))
        tk.Label(controls, textvariable=self.source_name, bg=SIDEBAR, fg=MUTED, wraplength=277,
                 justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 12))

        self._subhead(controls, "02 / AKILLI ANALİZ")
        tk.Checkbutton(controls,text="Görsel açıldığında otomatik analiz et",variable=self.auto_detect_var,
                       bg=SIDEBAR,fg=TEXT,selectcolor=PANEL,activebackground=SIDEBAR,
                       activeforeground=TEXT,font=("Segoe UI",9)).pack(anchor="w",pady=(0,10))
        self._subhead(controls, "03 / TESPİT AYARLARI")
        tk.Label(controls, text="Tespit yöntemi", bg=SIDEBAR, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(controls, textvariable=self.mode_var, values=list(MODES.values()),
                     state="readonly", width=34).pack(fill="x", pady=(3,10))
        tk.Label(controls,textvariable=self.ai_status_var,bg=SIDEBAR,
                 fg=GOOD,font=("Segoe UI",9),wraplength=295).pack(anchor="w",pady=(0,8))
        self.advanced_btn=self._button(controls,"⚙  GELİŞMİŞ PARAMETRELERİ GÖSTER",self._toggle_advanced)
        self.advanced_frame=tk.Frame(controls,bg=SIDEBAR)
        self._slider(self.advanced_frame, "Canny alt eşik", "canny_low", 0, 220)
        self._slider(self.advanced_frame, "Canny üst eşik", "canny_high", 1, 255)
        self._slider(self.advanced_frame, "ROI başlangıcı (%)", "roi_top", 35, 80)
        self._slider(self.advanced_frame, "Hough eşik", "hough_threshold", 5, 90)
        self._slider(self.advanced_frame, "En küçük çizgi (px)", "min_line_length", 10, 140)
        self._slider(self.advanced_frame, "Birleştirme aralığı", "max_line_gap", 5, 120)
        self._slider(self.advanced_frame, "Video yumuşatma (%)", "smoothing", 0, 95)

        self._subhead(controls, "04 / İŞLEMLER")
        self.btn_detect = self._button(controls, "ŞERİTLERİ TESPİT ET", self._detect, primary=True)
        self.btn_save = self._button(controls, "SONUCU KAYDET", self._save_result)
        self.btn_steps = self._button(controls, "ARA ADIMLARI KAYDET", self._save_steps)
        self.btn_label = self._button(controls, "EĞİTİM İÇİN ETİKETLE", self._label_current_image)
        self.btn_stop = self._button(controls, "VİDEOYU DURDUR", self._stop_video)
        self.btn_stop.configure(state="disabled")

        profile_frame=tk.Frame(main,bg=SIDEBAR,padx=18,pady=12)
        profile_frame.pack(fill="x",pady=(0,11))
        tk.Label(profile_frame,text="GÖRÜNTÜ PROFİLİ",bg=SIDEBAR,fg=MUTED,
                 font=("Segoe UI",9,"bold")).pack(anchor="w",pady=(0,7))
        row_profile=tk.Frame(profile_frame,bg=SIDEBAR)
        row_profile.pack(fill="x")
        tk.Label(row_profile,textvariable=self.scene_lighting,bg=PANEL,fg=GOOD,
                 font=("Segoe UI",11,"bold"),padx=14,pady=8).pack(side="left",padx=(0,9))
        tk.Label(row_profile,textvariable=self.scene_geometry,bg=PANEL,fg=BLUE,
                 font=("Segoe UI",11,"bold"),padx=14,pady=8).pack(side="left",padx=(0,9))
        tk.Label(row_profile,textvariable=self.scene_brightness,bg=SIDEBAR,fg=MUTED,
                 font=("Segoe UI",9),anchor="e").pack(side="right",fill="x",expand=True)
        tool_row = tk.Frame(main, bg=BG)
        tool_row.pack(fill="x", pady=(1, 9))
        tk.Label(tool_row, text="ŞERİT ANALİZİ  /  CANLI ÖNİZLEME", bg=BG, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self.stage_combo = ttk.Combobox(tool_row, textvariable=self.selected_stage, state="readonly", width=27,
                     values=["01 Orijinal", "02 Gri ton", "03 Bulaniklastirma", "04 Canny kenarlar",
                             "05 ROI maskesi", "06 ROI kenarlari", "07 Hough dogrulari", "08 Sonuc",
                             "09 AI olasilik haritasi", "10 AI sinif maskeleri"])
        self.stage_combo.pack(side="right")
        self.selected_stage.trace_add("write", lambda *_: self._refresh_stage())
        visual_frame = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        visual_frame.pack(fill="both", expand=True)
        self.image_panel = tk.Label(visual_frame, bg="#0B1728", fg=MUTED,
                                    text="Bir fotoğraf veya video açın", font=("Segoe UI", 13))
        self.image_panel.pack(fill="both", expand=True)
        self.image_panel.bind("<Configure>", lambda *_: self._render_preview())

        stats = tk.Frame(main, bg=BG)
        stats.pack(fill="x", pady=(12, 7))
        for i, (title, var) in enumerate((("SOL ŞERİT", self.metric_left),
                                           ("SAĞ ŞERİT", self.metric_right),
                                           ("MERKEZ SAPMASI", self.metric_offset),
                                           ("ÇİZGİ DESTEĞİ", self.metric_segments))):
            card = tk.Frame(stats, bg=PANEL, padx=13, pady=9)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8) if i < 3 else 0)
            stats.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w")
            tk.Label(card, textvariable=var, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(5, 0))
        tk.Label(main, textvariable=self.algorithm_method, bg=BG, fg=GOOD, anchor="w",
                 font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(5,0))
        feedback = tk.Frame(main, bg=SIDEBAR, padx=12, pady=8)
        feedback.pack(fill="x", pady=(8, 0))
        tk.Label(feedback, text="EĞİTİM GERİ BİLDİRİMİ", bg=SIDEBAR, fg=BLUE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(feedback, textvariable=self.feedback_status, bg=SIDEBAR, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w", wraplength=900).pack(fill="x", pady=(2, 6))
        feedback_actions = tk.Frame(feedback, bg=SIDEBAR)
        feedback_actions.pack(anchor="w")
        self.btn_feedback_ok = tk.Button(feedback_actions, text="✓ DOĞRU — EĞİTİME EKLE",
                                         command=self._confirm_detection, state="disabled",
                                         bg=GOOD, fg="#00172a", activebackground="#6df0d0", bd=0,
                                         font=("Segoe UI", 9, "bold"), padx=12, pady=6, cursor="hand2")
        self.btn_feedback_ok.pack(side="left", padx=(0, 7))
        self.btn_feedback_fix = tk.Button(feedback_actions, text="✎ HATALI — DOĞRUSUNU ÇİZ",
                                          command=self._review_incorrect, state="disabled",
                                          bg=PANEL, fg=TEXT, activebackground="#294767", bd=0,
                                          font=("Segoe UI", 9, "bold"), padx=12, pady=6, cursor="hand2")
        self.btn_feedback_fix.pack(side="left")
        self.btn_feedback_train = tk.Button(feedback_actions, text="ADAY MODELİ EĞİT",
                                            command=self._train_feedback_model,
                                            bg=BLUE, fg="#00172a", activebackground="#88dbff", bd=0,
                                            font=("Segoe UI", 9, "bold"), padx=12, pady=6, cursor="hand2")
        self.btn_feedback_train.pack(side="left", padx=(7, 0))
        self.progressbar = ttk.Progressbar(main, variable=self.progress_percent, maximum=100,
                                           style="Horizontal.TProgressbar")
        self.progressbar.pack(fill="x", pady=(10, 0))
        tk.Label(main, textvariable=self.progress_text, bg=BG, fg=MUTED, anchor="w",
                 font=("Segoe UI", 10)).pack(fill="x", pady=(4, 0))
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=25, pady=(0, 15))
        tk.Label(bottom, textvariable=self.status, bg=BG, fg=GOOD, anchor="w",
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(bottom, text="Eğitim ve araştırma amaçlıdır; gerçek araç kontrolünde kullanılmaz.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right")

    def _toggle_advanced(self):
        self.advanced_visible=not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill="x",pady=(0,12),after=self.advanced_btn)
        else:
            self.advanced_frame.pack_forget()
        self.advanced_btn.configure(text="⚙  GELİŞMİŞ PARAMETRELERİ GİZLE" if self.advanced_visible
                                    else "⚙  GELİŞMİŞ PARAMETRELERİ GÖSTER")

    def _analyze_loaded_scene(self, frame):
        try:
            p=analyze_scene(frame)
            self.scene_profile=p
            self.scene_lighting.set("☾  "+p.short_lighting if p.lighting=='gece' else "☀  "+p.short_lighting)
            self.scene_geometry.set("↝  "+p.short_geometry)
            self.scene_brightness.set(f"Işık endeksi {p.brightness:.0f}/255  ·  Viraj işareti: {p.curve_score:.1f}")
        except Exception:
            self.scene_profile=None
            self.scene_lighting.set("IŞIK: BELİRSİZ")
            self.scene_geometry.set("YOL: BELİRSİZ")
            self.scene_brightness.set("Yeterli görsel veri bulunamadı")

    def _subhead(self, parent, text):
        tk.Label(parent, text=text, bg=SIDEBAR, fg=BLUE,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 9))

    def _button(self, parent, label, command, primary=False):
        btn = tk.Button(parent, text=label, command=command,
                        bg=BLUE if primary else PANEL, fg="#00172a" if primary else TEXT,
                        activebackground="#88dbff", activeforeground="#00172a", bd=0,
                        font=("Segoe UI", 10, "bold"), padx=10, pady=9, cursor="hand2")
        btn.pack(fill="x", pady=(0, 5))
        return btn

    def _slider(self, parent, label, key, minimum, maximum):
        row = tk.Frame(parent, bg=SIDEBAR)
        row.pack(fill="x")
        value = self.settings_vars[key]
        tk.Label(row, text=label, fg=MUTED, bg=SIDEBAR,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(row, textvariable=value, fg=TEXT, bg=SIDEBAR,
                 font=("Segoe UI", 9, "bold")).pack(side="right")
        tk.Scale(parent, from_=minimum, to=maximum, variable=value, orient="horizontal",
                 bg=SIDEBAR, fg=TEXT, troughcolor=PANEL, highlightthickness=0,
                 activebackground=BLUE, bd=0, showvalue=False, sliderlength=14,
                 sliderrelief="flat", length=265).pack(fill="x", pady=(0, 1))

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
        name = filedialog.askopenfilename(title="Yol görüntüsü seç", filetypes=[
            ("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tüm dosyalar", "*.*")])
        if name:
            self._open_image(Path(name))

    def _select_video(self):
        if self.is_processing:
            messagebox.showwarning("İşlem sürüyor", "Önce mevcut video işlemini durdurun.")
            return
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
            ok, frame = cap.read() if cap.isOpened() else (False, None)
        finally:
            cap.release()
        if ok and frame is not None:
            self.current_display = frame
            self._analyze_loaded_scene(frame)
            self.selected_stage.set("01 Orijinal")
            self._render_preview()
        else:
            messagebox.showerror("Hata", "Video önizlemesi açılamadı.")
            self.current_file = None
            return
        self.status.set("Video seçildi. 'Şeritleri tespit et' ile işleyebilirsiniz.")
        self.progress_percent.set(0)
        self.progress_text.set("Video işlenmeyi bekliyor")
        self._clear_metrics()

    def _open_image(self, path: Path):
        try:
            self.current_image = read_image(path)
        except Exception as exc:
            messagebox.showerror("Dosya hatası", str(exc))
            return
        self.current_file = path
        self.current_detection = None
        self.source_name.set(path.name)
        self.current_display = self.current_image
        self._analyze_loaded_scene(self.current_image)
        self.selected_stage.set("01 Orijinal")
        self.progress_percent.set(0)
        self.progress_text.set("Görüntü işlenmeyi bekliyor")
        self.status.set(f"Görüntü açıldı: {path.name}. Tespit düğmesine basın.")
        self._clear_metrics()
        self._render_preview()
        if self.auto_detect_var.get():
            self.after(80, self._detect)

    def _clear_metrics(self):
        for v in (self.metric_left, self.metric_right, self.metric_offset, self.metric_segments):
            v.set("—")
        self._set_feedback_ready(False, "Özgün fotoğrafın tespitinden sonra sonucu doğru veya hatalı olarak işaretleyebilirsiniz.")

    def _set_feedback_ready(self, ready, message=None):
        state = "normal" if ready else "disabled"
        self.btn_feedback_ok.configure(state=state)
        self.btn_feedback_fix.configure(state=state)
        if message is not None:
            self.feedback_status.set(message)

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

    def _set_metrics(self, detection):
        marking_types=getattr(detection,"marking_types",(None,None))
        left_type,right_type=marking_types
        self.metric_left.set((left_type.upper()+" • BULUNDU") if detection.left and left_type else
                             ("BULUNDU" if detection.left else "BULUNAMADI"))
        self.metric_right.set((right_type.upper()+" • BULUNDU") if detection.right and right_type else
                              ("BULUNDU" if detection.right else "BULUNAMADI"))
        self.metric_offset.set(f"{detection.offset_px:+.1f} px" if detection.offset_px is not None else "ÖLÇÜLEMEDİ")
        self.metric_segments.set(str(detection.segment_count) if detection.segment_count else "—")
        self.algorithm_method.set("Kullanılan yöntem: " + detection.method)
        profile=getattr(detection,"scene_profile",None)
        if profile is not None:
            self.scene_profile=profile
            self.scene_lighting.set(("☾  " if profile.lighting=='gece' else "☀  ")+profile.short_lighting)
            self.scene_geometry.set("↝  "+profile.short_geometry)
            self.scene_brightness.set(f"Işık endeksi {profile.brightness:.0f}/255  ·  Viraj işareti: {profile.curve_score:.1f}")

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
                detection = InferenceEngine(settings, mode=self._mode()).detect(self.current_image)
            except Exception as exc:
                messagebox.showerror("İşlem hatası", str(exc))
                return
            self.current_detection = detection
            self.stage_combo.configure(values=list(detection.stages.keys()))
            self.selected_stage.set("08 Sonuc")
            self.current_display = detection.frame
            self._render_preview()
            self._set_metrics(detection)
            self._set_feedback_ready(True, "Sonucu kontrol edin: doğruysa eğitime ekleyin, hatalıysa doğru çizgileri çizin.")
            self.progress_percent.set(100)
            self.progress_text.set("Görüntü analizi tamamlandı")
            self.status.set("Tespit tamamlandı. Çıktıyı ve ara adımları kaydedebilirsiniz.")

    def _mode(self):
        return self.mode_keys.get(self.mode_var.get(), "auto")

    def _run_video(self, settings):
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
        self.progress_percent.set(0)
        self.progress_text.set("Video işleniyor...")
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

    def _stop_video(self):
        self.stop_event.set()
        self.status.set("Durdurma isteği gönderildi, mevcut kare tamamlanıyor.")

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
            messagebox.showinfo("Tespit yok", "Önce özgün bir fotoğrafta şerit tespiti yapın.")
            return
        try:
            sample = save_confirmed_detection(
                self.current_image, self.current_detection, APP_ROOT / "data" / "gercek", self.current_file.name)
        except ValueError as exc:
            messagebox.showinfo("Elle düzeltme gerekli", str(exc))
            self._review_incorrect()
            return
        count = style_sample_count(APP_ROOT / "data" / "gercek")
        self._set_feedback_ready(False, f"Onaylı örnek kaydedildi ({count} etiketli fotoğraf). "
                                  "Hatalı örneklerde 'doğrusunu çiz' seçeneğini kullanın.")
        self.status.set(f"Onaylı eğitim örneği yerelde kaydedildi: {sample['image'].name}")

    def _review_incorrect(self):
        self._launch_labeler(correction=True)

    def _train_feedback_model(self):
        """Train a local candidate; never overwrite the current model silently."""
        count=style_sample_count(APP_ROOT / "data" / "gercek")
        if count<24:
            messagebox.showinfo("Daha fazla etiket gerekli",
                                f"Aday eğitim için en az 24 doğru etiketli fotoğraf önerilir. Şu anda: {count}.")
            return
        if not messagebox.askyesno(
            "Aday modeli eğit",
            f"{count} yerel etiketle aday model eğitilecek. Bu işlem sürebilir ve mevcut modeli değiştirmez. Devam edilsin mi?"
        ):
            return
        candidate=APP_ROOT / "modeller" / "serit_unet_geri_bildirim_candidate.pt"
        try:
            subprocess.Popen([
                sys.executable, str(APP_ROOT / "egit.py"), "--style-aware", "--synthetic", "240",
                "--epochs", "15", "--batch", "4", "--out", str(candidate),
            ], cwd=str(APP_ROOT))
        except OSError as exc:
            messagebox.showerror("Eğitim başlatılamadı", str(exc))
            return
        self.feedback_status.set("Aday model eğitimi arka planda başladı. Mevcut model korunuyor; aday önce test edilmelidir.")
        self.status.set("Geri bildirim aday modeli eğitiliyor. Uygulama kullanılmaya devam edebilir.")

    def _label_current_image(self):
        self._launch_labeler(correction=False)

    def _launch_labeler(self, correction: bool):
        """Özgün fotoğraf için dört sınıflı etiketleme penceresini aç."""
        if self.current_file is None or self.current_image is None:
            messagebox.showinfo("Fotoğraf seçin", "Önce etiketlemek istediğiniz özgün yol fotoğrafını açın.")
            return
        if self.current_file.suffix.lower() in VIDEO_EXTENSIONS:
            messagebox.showinfo("Video karesi", "Videodan önce bir kareyi görsel olarak kaydedip sonra o görseli açın.")
            return
        if not correction and not messagebox.askyesno(
            "Eğitim etiketi",
            "Açılan pencerede 1=sol düz, 2=sol kesik, 3=sağ düz, 4=sağ kesik, 5=aradaki alan seçilir.\n"
            "Her çizgi için en az iki nokta koyun. Özgün görsel yalnızca yerel eğitim klasörüne kaydedilecek."
        ):
            return
        name=f"{self.current_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            subprocess.Popen([
                sys.executable, str(APP_ROOT / "etiketle.py"), "--image", str(self.current_file),
                "--name", name, "--outdir", str(APP_ROOT / "data" / "gercek"), "--style-aware",
            ], cwd=str(APP_ROOT))
        except OSError as exc:
            messagebox.showerror("Etiketleme açılamadı", str(exc))
            return
        self._set_feedback_ready(False, "Etiketleme penceresinde doğru çizgileri işaretleyip S ile kaydedin.")
        self.status.set("Etiketleme penceresi açıldı. Kaydedilen özgün görsel yalnızca yerel eğitim verisine eklenir.")

    def _on_close(self):
        if self.is_processing:
            messagebox.showwarning("Video işleniyor", "Önce 'Videoyu durdur' düğmesine basıp işlemin bitmesini bekleyin.")
            self.stop_event.set()
            return
        self.destroy()


def main():
    LaneApp().mainloop()
