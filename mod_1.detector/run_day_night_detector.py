#!/usr/bin/env python3
"""Отдельное приложение для проверки только Day -> Night-определения.

Запуск:
    python run_day_night_detector.py

Трекер, фильтр движения и основной reflector_tracker не импортируются.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from day_night_reflector_detector import (
    DayNightReflectorDetector,
    DetectionBatch,
    DetectorSettings,
    Rect,
)
from day_night_reflector_detector.hikvision_camera import (
    HikvisionCameraRig,
    HikvisionISAPI,
    RTSPCamera,
    credentials_from_rtsp,
    isapi_url_from_rtsp,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

APP_TITLE = "Day→Night Reflector Detector — отдельный модуль"
CONFIG_PATH = Path(__file__).with_name("day_night_detector_config.json")
class DetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1420x900")
        self.root.minsize(1050, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.camera: Optional[RTSPCamera] = None
        self.isapi: Optional[HikvisionISAPI] = None
        self.worker: Optional[threading.Thread] = None
        self.events: queue.Queue = queue.Queue()
        self.current_frame: Optional[np.ndarray] = None
        self.result_batch: Optional[DetectionBatch] = None
        self.result_preview_active = False
        self.rois: List[Rect] = []
        self.selection_active = False
        self.selection_start: Optional[Tuple[float, float]] = None
        self.selection_current: Optional[Tuple[float, float]] = None

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_origin: Optional[Tuple[int, int, float, float]] = None
        self.display_transform = (1.0, 0.0, 0.0)
        self.frame_for_selection: Optional[np.ndarray] = None
        self.photo: Optional[ImageTk.PhotoImage] = None

        self._make_ui()
        self._load_config()
        self.root.after(50, self._poll_events)
        self.root.after(80, self._refresh_video)

    def _make_ui(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(outer, padding=8)
        right = ttk.Frame(outer)
        outer.add(left, weight=0)
        outer.add(right, weight=1)

        source = ttk.LabelFrame(left, text="Камера", padding=6)
        source.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(source, text="RTSP URL:").pack(anchor=tk.W)
        self.rtsp_var = tk.StringVar()
        ttk.Entry(source, textvariable=self.rtsp_var, width=48).pack(fill=tk.X)
        ttk.Label(source, text="ISAPI URL:").pack(anchor=tk.W, pady=(5, 0))
        api_row = ttk.Frame(source)
        api_row.pack(fill=tk.X)
        self.api_var = tk.StringVar()
        ttk.Entry(api_row, textvariable=self.api_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(api_row, text="↻", width=3, command=self._api_from_rtsp).pack(
            side=tk.LEFT, padx=(3, 0)
        )
        auth = ttk.Frame(source)
        auth.pack(fill=tk.X, pady=(5, 0))
        self.user_var = tk.StringVar()
        self.password_var = tk.StringVar()
        ttk.Label(auth, text="Логин:").grid(row=0, column=0, sticky="w")
        ttk.Entry(auth, textvariable=self.user_var, width=13).grid(
            row=0, column=1, padx=(3, 8)
        )
        ttk.Label(auth, text="Пароль:").grid(row=0, column=2, sticky="w")
        ttk.Entry(auth, textvariable=self.password_var, show="•", width=15).grid(
            row=0, column=3, padx=(3, 0), sticky="ew"
        )
        auth.columnconfigure(3, weight=1)
        mode = ttk.Frame(source)
        mode.pack(fill=tk.X, pady=(5, 0))
        self.channel_var = tk.IntVar(value=1)
        ttk.Label(mode, text="Канал:").pack(side=tk.LEFT)
        ttk.Spinbox(mode, from_=1, to=64, textvariable=self.channel_var, width=5).pack(
            side=tk.LEFT, padx=(3, 8)
        )
        ttk.Button(mode, text="Проверить ISAPI", command=self.test_isapi).pack(
            side=tk.RIGHT
        )
        connect = ttk.Frame(source)
        connect.pack(fill=tk.X, pady=(6, 0))
        self.connect_button = ttk.Button(
            connect, text="Подключить RTSP", command=self.toggle_connection
        )
        self.connect_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(connect, text="Day", command=lambda: self.set_mode("day")).pack(
            side=tk.LEFT, padx=(3, 0)
        )
        ttk.Button(connect, text="Night", command=lambda: self.set_mode("night")).pack(
            side=tk.LEFT, padx=(3, 0)
        )
        self.connection_label = ttk.Label(source, text="Не подключено", foreground="red")
        self.connection_label.pack(anchor=tk.W, pady=(4, 0))

        roi_box = ttk.LabelFrame(left, text="Строгие области P1…Pn", padding=6)
        roi_box.pack(fill=tk.X, pady=6)
        roi_row = ttk.Frame(roi_box)
        roi_row.pack(fill=tk.X)
        self.expected_var = tk.IntVar(value=2)
        ttk.Label(roi_row, text="Количество:").pack(side=tk.LEFT)
        ttk.Spinbox(
            roi_row, from_=1, to=20, textvariable=self.expected_var, width=5
        ).pack(side=tk.LEFT, padx=(3, 8))
        self.select_button = ttk.Button(
            roi_row, text="Задать рамки", command=self.toggle_roi_selection
        )
        self.select_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(roi_row, text="Очистить", command=self.clear_rois).pack(
            side=tk.LEFT, padx=(3, 0)
        )
        self.roi_label = ttk.Label(roi_box, text="Области: 0/2")
        self.roi_label.pack(anchor=tk.W, pady=(4, 0))
        zoom_row = ttk.Frame(roi_box)
        zoom_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(zoom_row, text="Масштаб:").pack(side=tk.LEFT)
        ttk.Button(zoom_row, text="−", width=3, command=lambda: self._zoom(0.8)).pack(
            side=tk.LEFT, padx=(3, 1)
        )
        self.zoom_label = ttk.Label(zoom_row, text="1.00×", width=7, anchor="center")
        self.zoom_label.pack(side=tk.LEFT)
        ttk.Button(zoom_row, text="+", width=3, command=lambda: self._zoom(1.25)).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(zoom_row, text="Сброс", command=self._reset_view).pack(side=tk.RIGHT)

        settings = ttk.LabelFrame(left, text="Критерии Day→Night", padding=6)
        settings.pack(fill=tk.X, pady=6)
        self.setting_vars: Dict[str, tk.Variable] = {}
        fields = [
            ("day_black_max", "Day: максимум чёрного", 85, int),
            ("night_bright_min", "Night: минимум яркого", 155, int),
            ("min_positive_gain", "Мин. прирост Night−Day", 65, int),
            ("min_area", "Мин. площадь, px", 3, int),
            ("max_area", "Макс. площадь, px", 1400, int),
            ("blur_sigma", "Размытие sigma", 0.55, float),
            ("day_match_radius", "Допуск Day/Night, px", 5, int),
            ("repeat_max_distance", "Повторяемость, px", 12.0, float),
            ("center_power", "Вес яркого ядра", 2.2, float),
            ("minimum_score", "Мин. итоговый балл", 0.42, float),
            ("minimum_diamond_score", "Мин. ромбовидность", 0.0, float),
        ]
        for row, (name, label, default, value_type) in enumerate(fields):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=1)
            variable: tk.Variable
            variable = tk.IntVar(value=default) if value_type is int else tk.DoubleVar(value=default)
            self.setting_vars[name] = variable
            ttk.Entry(settings, textvariable=variable, width=9).grid(
                row=row, column=1, sticky="e", padx=(8, 0), pady=1
            )
        settings.columnconfigure(0, weight=1)

        capture = ttk.LabelFrame(left, text="Двойная проверка", padding=6)
        capture.pack(fill=tk.X, pady=6)
        capture_row = ttk.Frame(capture)
        capture_row.pack(fill=tk.X)
        self.samples_var = tk.IntVar(value=5)
        self.settle_var = tk.DoubleVar(value=2.0)
        ttk.Label(capture_row, text="Кадров:").pack(side=tk.LEFT)
        ttk.Spinbox(
            capture_row, from_=3, to=30, textvariable=self.samples_var, width=5
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(capture_row, text="Стабилизация, с:").pack(side=tk.LEFT)
        ttk.Spinbox(
            capture_row,
            from_=0.2,
            to=15.0,
            increment=0.2,
            textvariable=self.settle_var,
            width=6,
        ).pack(side=tk.RIGHT)
        self.scan_button = ttk.Button(
            capture,
            text="Найти: Day→Night ×2",
            command=self.start_scan,
        )
        self.scan_button.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(capture, text="Сохранить последний результат", command=self.save_result).pack(
            fill=tk.X, pady=2
        )
        self.progress_label = ttk.Label(
            capture, text="Готово", wraplength=360, foreground="gray"
        )
        self.progress_label.pack(fill=tk.X, pady=(4, 0))

        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom(1.15, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom(1 / 1.15, event.x, event.y))

    def _api_from_rtsp(self) -> None:
        self.api_var.set(isapi_url_from_rtsp(self.rtsp_var.get()))

    def _credentials(self) -> Tuple[str, str]:
        rtsp_user, rtsp_password = credentials_from_rtsp(self.rtsp_var.get())
        return self.user_var.get().strip() or rtsp_user, self.password_var.get() or rtsp_password

    def _make_isapi(self) -> HikvisionISAPI:
        user, password = self._credentials()
        api_url = self.api_var.get().strip() or isapi_url_from_rtsp(self.rtsp_var.get())
        return HikvisionISAPI(api_url, user, password, self.channel_var.get())

    def toggle_connection(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
            self.connection_label.config(text="Не подключено", foreground="red")
            self.connect_button.config(text="Подключить RTSP")
            self.result_preview_active = False
            return
        url = self.rtsp_var.get().strip()
        if not url:
            messagebox.showerror("RTSP", "Введите RTSP URL")
            return
        self._set_busy(True, "Подключение к RTSP…")

        def work():
            camera = RTSPCamera(url)
            camera.connect()
            frame, _ = camera.wait_next_frame(timeout=12)
            return camera, frame

        self._run_worker(work, "connected")

    def test_isapi(self) -> None:
        self._set_busy(True, "Проверка ISAPI…")

        def work():
            control = self._make_isapi()
            info = control.get_device_info()
            mode, _, _ = control.get_ircut()
            return control, info, mode

        self._run_worker(work, "isapi")

    def set_mode(self, mode: str) -> None:
        self._set_busy(True, f"Переключение в {mode}…")

        def work():
            control = self.isapi or self._make_isapi()
            control.set_ircut(mode)
            return control, mode

        self._run_worker(work, "mode")

    def toggle_roi_selection(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("Области", "Сначала подключите RTSP и дождитесь кадра")
            return
        self.selection_active = not self.selection_active
        self.selection_start = None
        self.selection_current = None
        if self.selection_active:
            if self.result_preview_active and self.camera is not None:
                latest = self.camera.get_latest_frame()
                if latest is not None:
                    self.current_frame = latest
            self.result_preview_active = False
            self.result_batch = None
            self.frame_for_selection = self.current_frame.copy()
            self.rois = []
            self.select_button.config(text="Отменить выбор")
            self.progress_label.config(
                text="Нарисуйте P1…Pn левой кнопкой прямо в текущем окне",
                foreground="#9a6500",
            )
        else:
            self.frame_for_selection = None
            self.select_button.config(text="Задать рамки")
        self._update_roi_label()

    def clear_rois(self) -> None:
        self.rois = []
        self.result_batch = None
        self.result_preview_active = False
        self.selection_active = False
        self.frame_for_selection = None
        self.select_button.config(text="Задать рамки")
        self._update_roi_label()

    def _on_left_press(self, event) -> None:
        if not self.selection_active or self.current_frame is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is not None:
            self.selection_start = point
            self.selection_current = point

    def _on_left_motion(self, event) -> None:
        if self.selection_start is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is not None:
            self.selection_current = point

    def _on_left_release(self, event) -> None:
        if self.selection_start is None or self.current_frame is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        start = self.selection_start
        self.selection_start = None
        self.selection_current = None
        if point is None:
            return
        x1, y1 = int(min(start[0], point[0])), int(min(start[1], point[1]))
        x2, y2 = int(max(start[0], point[0])), int(max(start[1], point[1]))
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            self.rois.append((x1, y1, x2 - x1, y2 - y1))
        if len(self.rois) >= max(1, int(self.expected_var.get())):
            self.selection_active = False
            self.frame_for_selection = None
            self.select_button.config(text="Задать рамки")
            self.progress_label.config(text="Области заданы", foreground="green")
        self._update_roi_label()

    def _on_pan_start(self, event) -> None:
        self.drag_origin = (event.x, event.y, self.pan_x, self.pan_y)

    def _on_pan_move(self, event) -> None:
        if self.drag_origin is None:
            return
        x, y, old_x, old_y = self.drag_origin
        self.pan_x = old_x + event.x - x
        self.pan_y = old_y + event.y - y

    def _on_mousewheel(self, event) -> None:
        self._zoom(1.15 if event.delta > 0 else 1 / 1.15, event.x, event.y)

    def _zoom(self, factor: float, canvas_x: Optional[int] = None, canvas_y: Optional[int] = None) -> None:
        old = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 0.25, 8.0))
        if canvas_x is not None and canvas_y is not None and old > 0:
            scale, offset_x, offset_y = self.display_transform
            image_x = (canvas_x - offset_x) / max(scale, 1e-9)
            image_y = (canvas_y - offset_y) / max(scale, 1e-9)
            # Точное удержание курсора достигается при следующей отрисовке.
            self.pan_x += (canvas_x - (image_x * scale + offset_x))
            self.pan_y += (canvas_y - (image_y * scale + offset_y))
        self.zoom_label.config(text=f"{self.zoom:.2f}×")

    def _reset_view(self) -> None:
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self.zoom_label.config(text="1.00×")

    def _canvas_to_image(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self.current_frame is None:
            return None
        scale, offset_x, offset_y = self.display_transform
        image_x = (x - offset_x) / max(scale, 1e-9)
        image_y = (y - offset_y) / max(scale, 1e-9)
        height, width = self.current_frame.shape[:2]
        if 0 <= image_x < width and 0 <= image_y < height:
            return image_x, image_y
        return None

    def _update_roi_label(self) -> None:
        expected = max(1, int(self.expected_var.get()))
        self.roi_label.config(text=f"Области: {len(self.rois)}/{expected}")

    def _detector_settings(self) -> DetectorSettings:
        values = {name: variable.get() for name, variable in self.setting_vars.items()}
        settings = DetectorSettings(**values)
        settings.validate()
        return settings

    def start_scan(self) -> None:
        if self.camera is None or not self.camera.is_connected:
            messagebox.showerror("Поиск", "RTSP-камера не подключена")
            return
        if len(self.rois) != max(1, int(self.expected_var.get())):
            messagebox.showerror("Поиск", "Задайте ровно указанное количество областей")
            return
        try:
            settings = self._detector_settings()
            control = self.isapi or self._make_isapi()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return
        self._set_busy(True, "Запуск двух циклов Day→Night…")
        self.result_preview_active = False
        self.frame_for_selection = None
        self.result_batch = None
        rois = list(self.rois)
        camera = self.camera

        def progress(text: str) -> None:
            self.events.put(("progress", text))

        def work():
            rig = HikvisionCameraRig(camera, control)
            capture = rig.capture_day_night_cycles(
                cycles=2,
                samples_per_state=int(self.samples_var.get()),
                settle_seconds=float(self.settle_var.get()),
                leave_in_night=True,
                progress=progress,
            )
            detector = DayNightReflectorDetector(settings)
            batch = detector.analyze(capture.day_cycles, capture.night_cycles, rois)
            return control, batch

        self._run_worker(work, "scan")

    def save_result(self) -> None:
        if self.result_batch is None:
            messagebox.showinfo("Сохранение", "Сначала выполните Day→Night-поиск")
            return
        selected = filedialog.askdirectory(title="Каталог для диагностического результата")
        if not selected:
            return
        target = Path(selected) / f"day_night_scan_{datetime.now():%Y%m%d_%H%M%S}"
        self.result_batch.save(target)
        self.progress_label.config(text=f"Сохранено: {target}", foreground="green")

    def _set_busy(self, busy: bool, text: Optional[str] = None) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.scan_button.config(state=state)
        self.connect_button.config(state=state)
        if text:
            self.progress_label.config(text=text, foreground="#9a6500")

    def _run_worker(self, function, event_name: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning("Операция", "Дождитесь завершения текущей операции")
            return

        def target():
            try:
                self.events.put((event_name, function()))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _poll_events(self) -> None:
        try:
            while True:
                name, payload = self.events.get_nowait()
                if name == "progress":
                    self.progress_label.config(text=str(payload), foreground="#9a6500")
                elif name == "connected":
                    self.camera, self.current_frame = payload
                    self.result_preview_active = False
                    self.frame_for_selection = None
                    self.connect_button.config(text="Отключить RTSP")
                    self.connection_label.config(text="RTSP подключён", foreground="green")
                    self._set_busy(False, "Кадр получен. Задайте строгие рамки Pn.")
                elif name == "isapi":
                    self.isapi, info, mode = payload
                    model = info.get("model") or info.get("deviceName") or "Hikvision"
                    self._set_busy(False, f"ISAPI: {model}; режим {mode}")
                elif name == "mode":
                    self.isapi, mode = payload
                    self._set_busy(False, f"Камера переключена в {mode}")
                elif name == "scan":
                    self.isapi, self.result_batch = payload
                    self.current_frame = self.result_batch.annotated_frame()
                    self.result_preview_active = True
                    found = sum(item.found for item in self.result_batch.results)
                    total = len(self.result_batch.results)
                    self._set_busy(False, f"Готово: найдено {found}/{total}; камера оставлена в Night")
                    self._auto_save_last_result()
                elif name == "error":
                    self._set_busy(False, f"Ошибка: {payload}")
                    messagebox.showerror("Ошибка", str(payload))
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _auto_save_last_result(self) -> None:
        if self.result_batch is None:
            return
        target = Path(__file__).with_name("day_night_diagnostics") / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.result_batch.save(target)

    def _refresh_video(self) -> None:
        if (
            self.camera is not None
            and self.camera.is_connected
            and not self.result_preview_active
            and not self.selection_active
        ):
            latest = self.camera.get_latest_frame()
            if latest is not None:
                self.current_frame = latest
        if self.current_frame is not None:
            self._draw_frame()
        self.root.after(80, self._refresh_video)

    def _draw_frame(self) -> None:
        frame = self.current_frame.copy()
        for index, (x, y, w, h) in enumerate(self.rois, start=1):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 190, 0), 2)
            cv2.putText(
                frame,
                f"P{index}",
                (x + 2, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 190, 0),
                2,
                cv2.LINE_AA,
            )
        if self.selection_start is not None and self.selection_current is not None:
            x1, y1 = [int(value) for value in self.selection_start]
            x2, y2 = [int(value) for value in self.selection_current]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        height, width = frame.shape[:2]
        fit = min(canvas_w / width, canvas_h / height)
        scale = max(0.05, fit * self.zoom)
        display_w = max(1, int(round(width * scale)))
        display_h = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        offset_x = (canvas_w - display_w) / 2.0 + self.pan_x
        offset_y = (canvas_h - display_h) / 2.0 + self.pan_y
        self.display_transform = (scale, offset_x, offset_y)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor=tk.NW)

    def _config_data(self) -> Dict:
        return {
            "rtsp_url": self.rtsp_var.get(),
            "isapi_url": self.api_var.get(),
            "username": self.user_var.get(),
            # Отдельное поле пароля не записывается. Если реквизиты находятся
            # непосредственно в RTSP URL, строка URL сохранится как введена.
            "channel": self.channel_var.get(),
            "expected_reflectors": self.expected_var.get(),
            "samples_per_state": self.samples_var.get(),
            "settle_seconds": self.settle_var.get(),
            "regions": [list(roi) for roi in self.rois],
            "detector": asdict(self._detector_settings()),
        }

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
            self.rtsp_var.set(str(data.get("rtsp_url", "")))
            self.api_var.set(str(data.get("isapi_url", "")))
            self.user_var.set(str(data.get("username", "")))
            self.channel_var.set(int(data.get("channel", 1)))
            self.expected_var.set(int(data.get("expected_reflectors", 2)))
            self.samples_var.set(int(data.get("samples_per_state", 5)))
            self.settle_var.set(float(data.get("settle_seconds", 2.0)))
            self.rois = [tuple(int(value) for value in roi) for roi in data.get("regions", [])]
            detector = data.get("detector", {})
            for name, variable in self.setting_vars.items():
                if name in detector:
                    variable.set(detector[name])
            self._update_roi_label()
        except Exception as exc:
            logging.warning("Не удалось загрузить конфигурацию: %s", exc)

    def _save_config(self) -> None:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as file:
                json.dump(self._config_data(), file, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Не удалось сохранить конфигурацию: %s", exc)

    def close(self) -> None:
        self._save_config()
        if self.camera is not None:
            self.camera.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
