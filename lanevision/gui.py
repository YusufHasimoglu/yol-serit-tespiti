"""Tkinter ile Türkçe masaüstü arayüz. OpenCV worker yalnızca kuyruk üzerinden UI'yi günceller."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .core import Detection, LaneSettings
from .engine import InferenceEngine, MODES
from .processing import VIDEO_EXTENSIONS, process_video, read_image, save_stages, write_image

APP_ROOT = Path(__file__).resolve().parent.parent
BG = "#0b1425"
SIDEBAR = "#131f33"
PANEL = "#182840"
BLUE = "#35b5ff"
TEXT = "#e9f1ff"
MUTED = "#a7bbd3"
GOOD = "#2ed8a1"


class LaneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ŞERİT GÖRÜ  |  Yol Çizgisi Tespit Sistemi")
        self.geometry("1190x765")
        self.minsize(950, 650)
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
        self.status = tk.StringVar(value="Hazır. Örnek görüntüyle başlayabilir veya kendi dosyanızı açabilirsiniz.")
        self.metric_left = tk.StringVar(value="—")
        self.metric_right = tk.StringVar(value="—")
        self.metric_offset = tk.StringVar(value="—")
        self.metric_segments = tk.StringVar(value="—")
        self.source_name = tk.StringVar(value="Dosya seçilmedi")
        self.progress_percent = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="İşlem bekleniyor")
        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._process_messages)
        self.after(200, lambda: self._load_example_if_exists(force=False))

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#dfeaf8", background="#dfeaf8", padding=6)
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=BLUE, borderwidth=0)

    def _build_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=25, pady=(18, 10))
        tk.Label(header, text="ŞERİT GÖRÜ", bg=BG, fg=BLUE, font=("Segoe UI", 24, "bold")).pack(side="left")
        tk.Label(header, text="  /  Yol Görüntülerinden Şerit Tespiti", bg=BG, fg=MUTED,
                 font=("Segoe UI", 13)).pack(side="left", pady=(8, 0))
        tk.Label(header, text="OpenCV + Eğrisel Takip + U-Net", bg=BG, fg=GOOD,
                 font=("Segoe UI", 10, "bold")).pack(side="right", pady=(10, 0))

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        # Dar ekranlarda son düğmelerin kaybolmaması için kaydırılabilir sol panel.
        sidebar_holder = tk.Frame(container, bg=SIDEBAR, width=315)
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

        self._subhead(controls, "02 / ALGORİTMA AYARLARI")
        tk.Label(controls, text="Tespit yöntemi", bg=SIDEBAR, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(controls, textvariable=self.mode_var, values=list(MODES.values()),
                     state="readonly", width=34).pack(fill="x", pady=(3,10))
        tk.Label(controls, text="AI modu: pip install -r requirements-ai.txt", bg=SIDEBAR,
                 fg=MUTED, font=("Segoe UI", 8), wraplength=270).pack(anchor="w", pady=(0,8))
        self._slider(controls, "Canny alt eşik", "canny_low", 0, 220)
        self._slider(controls, "Canny üst eşik", "canny_high", 1, 255)
        self._slider(controls, "ROI başlangıcı (%)", "roi_top", 35, 80)
        self._slider(controls, "Hough eşik", "hough_threshold", 5, 90)
        self._slider(controls, "En küçük çizgi (px)", "min_line_length", 10, 140)
        self._slider(controls, "Birleştirme aralığı", "max_line_gap", 5, 120)
        self._slider(controls, "Video yumuşatma (%)", "smoothing", 0, 95)

        self._subhead(controls, "03 / İŞLEMLER")
        self.btn_detect = self._button(controls, "ŞERİTLERİ TESPİT ET", self._detect, primary=True)
        self.btn_save = self._button(controls, "SONUCU KAYDET", self._save_result)
        self.btn_steps = self._button(controls, "ARA ADIMLARI KAYDET", self._save_steps)
        self.btn_stop = self._button(controls, "VİDEOYU DURDUR", self._stop_video)
        self.btn_stop.configure(state="disabled")

        tool_row = tk.Frame(main, bg=BG)
        tool_row.pack(fill="x", pady=(1, 9))
        tk.Label(tool_row, text="GÖRSEL ÖNİZLEME", bg=BG, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self.stage_combo = ttk.Combobox(tool_row, textvariable=self.selected_stage, state="readonly", width=27,
                     values=["01 Orijinal", "02 Gri ton", "03 Bulaniklastirma", "04 Canny kenarlar",
                             "05 ROI maskesi", "06 ROI kenarlari", "07 Hough dogrulari", "08 Sonuc",
                             "09 AI olasilik haritasi", "10 AI sinif maskeleri"])
        self.stage_combo.pack(side="right")
        self.selected_stage.trace_add("write", lambda *_: self._refresh_stage())
        visual_frame = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        visual_frame.pack(fill="both", expand=True)
        self.image_panel = tk.Label(visual_frame, bg="#0c1627", fg=MUTED,
                                    text="Bir fotoğraf veya video açın", font=("Segoe UI", 13))
        self.image_panel.pack(fill="both", expand=True)
        self.image_panel.bind("<Configure>", lambda *_: self._render_preview())

        stats = tk.Frame(main, bg=BG)
        stats.pack(fill="x", pady=(12, 7))
        for i, (title, var) in enumerate((("SOL ŞERİT", self.metric_left),
                                           ("SAĞ ŞERİT", self.metric_right),
                                           ("MERKEZ SAPMASI", self.metric_offset),
                                           ("HOUGH PARÇASI", self.metric_segments))):
            card = tk.Frame(stats, bg=PANEL, padx=13, pady=9)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8) if i < 3 else 0)
            stats.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w")
            tk.Label(card, textvariable=var, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(5, 0))
        tk.Label(main, textvariable=self.algorithm_method, bg=BG, fg=GOOD, anchor="w",
                 font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(5,0))
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
        self.selected_stage.set("01 Orijinal")
        self.progress_percent.set(0)
        self.progress_text.set("Görüntü işlenmeyi bekliyor")
        self.status.set(f"Görüntü açıldı: {path.name}. Tespit düğmesine basın.")
        self._clear_metrics()
        self._render_preview()

    def _clear_metrics(self):
        for v in (self.metric_left, self.metric_right, self.metric_offset, self.metric_segments):
            v.set("—")

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
        self.metric_left.set("BULUNDU" if detection.left else "BULUNAMADI")
        self.metric_right.set("BULUNDU" if detection.right else "BULUNAMADI")
        self.metric_offset.set(f"{detection.offset_px:+.1f} px" if detection.offset_px is not None else "ÖLÇÜLEMEDİ")
        self.metric_segments.set(str(detection.segment_count) if detection.segment_count else "—")
        self.algorithm_method.set("Kullanılan yöntem: " + detection.method)

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

    def _on_close(self):
        if self.is_processing:
            messagebox.showwarning("Video işleniyor", "Önce 'Videoyu durdur' düğmesine basıp işlemin bitmesini bekleyin.")
            self.stop_event.set()
            return
        self.destroy()


def main():
    LaneApp().mainloop()