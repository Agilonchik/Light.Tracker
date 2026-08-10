import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import ssl
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import xml.etree.ElementTree as ET

from PIL import Image, ImageGrab, ImageTk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Rect = Tuple[int, int, int, int]
Point = Tuple[float, float]

PRESET_FIELDS = (
    "min_area",
    "max_area",
    "circularity_threshold",
    "brightness_threshold",
    "contrast_threshold",
    "blur_sigma",
    "expected_reflectors",
    "adaptive_threshold",
    "brightness_percentile",
    "merge_radius",
    "center_power",
    "smoothing_alpha",
    "roi_expand_step",
    "roi_max_scale",
    "lost_hold_frames",
    "max_jump",
)

# Эти параметры можно переопределить отдельно для каждой области P1...Pn.
# Параметры сопровождения остаются общими, чтобы геометрия всех точек
# фильтровалась одинаково.
REGION_DETECTION_FIELDS = (
    "min_area",
    "max_area",
    "circularity_threshold",
    "brightness_threshold",
    "contrast_threshold",
    "blur_sigma",
    "adaptive_threshold",
    "brightness_percentile",
    "merge_radius",
    "center_power",
)

REGION_FIELD_META = {
    "min_area": ("Мин. площадь, px", int, 1.0, 100000.0),
    "max_area": ("Макс. площадь, px", int, 1.0, 100000.0),
    "circularity_threshold": ("Ожидаемая округлость", float, 0.05, 1.0),
    "brightness_threshold": ("Мин. яркость", int, 0.0, 255.0),
    "contrast_threshold": ("Мин. локальный контраст", int, 1.0, 255.0),
    "blur_sigma": ("Размытие sigma", float, 0.0, 20.0),
    "brightness_percentile": ("Процентиль яркости", float, 80.0, 99.95),
    "merge_radius": ("Радиус объединения, px", int, 0.0, 100.0),
    "center_power": ("Вес яркого ядра центра", float, 1.0, 5.0),
}

BUILTIN_PRESETS = {
    "Базовый": {
        "min_area": 5,
        "max_area": 257,
        "circularity_threshold": 0.30,
        "brightness_threshold": 235,
        "contrast_threshold": 45,
        "blur_sigma": 0.49,
        "expected_reflectors": 2,
        "adaptive_threshold": False,
        "brightness_percentile": 98.5,
        "merge_radius": 3,
        "center_power": 3.50,
        "smoothing_alpha": 0.20,
        "roi_expand_step": 1,
        "roi_max_scale": 2.0,
        "lost_hold_frames": 75,
        "max_jump": 9.0,
    },
    "Фиксированный точный": {
        "min_area": 4,
        "max_area": 220,
        "circularity_threshold": 0.25,
        "brightness_threshold": 238,
        "contrast_threshold": 50,
        "blur_sigma": 0.45,
        "expected_reflectors": 2,
        "adaptive_threshold": False,
        "brightness_percentile": 99.0,
        "merge_radius": 2,
        "center_power": 4.0,
        "smoothing_alpha": 0.15,
        "roi_expand_step": 1,
        "roi_max_scale": 2.0,
        "lost_hold_frames": 120,
        "max_jump": 5.0,
    },
    "Широкий блик": {
        "min_area": 5,
        "max_area": 700,
        "circularity_threshold": 0.20,
        "brightness_threshold": 215,
        "contrast_threshold": 35,
        "blur_sigma": 0.80,
        "expected_reflectors": 2,
        "adaptive_threshold": False,
        "brightness_percentile": 98.5,
        "merge_radius": 7,
        "center_power": 1.50,
        "smoothing_alpha": 0.20,
        "roi_expand_step": 1,
        "roi_max_scale": 2.0,
        "lost_hold_frames": 90,
        "max_jump": 10.0,
    },
}


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Не найден исполняемый файл {name}. "
            "Установите FFmpeg и добавьте его в PATH."
        )
    return executable


def run_ffmpeg_command(
    command: List[str], timeout: Optional[float] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def clip_rect(rect: Rect, frame_shape: Tuple[int, ...]) -> Optional[Rect]:
    x, y, w, h = [int(round(v)) for v in rect]
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w, x))
    y1 = max(0, min(frame_h, y))
    x2 = max(0, min(frame_w, x + max(0, w)))
    y2 = max(0, min(frame_h, y + max(0, h)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2 - x1, y2 - y1


def rect_distance(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(ax - (bx + bw), bx - (ax + aw), 0)
    dy = max(ay - (by + bh), by - (ay + ah), 0)
    return float(np.hypot(dx, dy))


@dataclass
class PeakDetection:
    position: Point
    radius: float
    confidence: float
    area: float
    circularity: float
    bbox: Rect
    threshold: float
    max_intensity: float


@dataclass
class ReflectorPoint:
    id: int
    base_region: Rect
    position: Point
    radius: float = 5.0
    confidence: float = 0.0
    last_seen: float = 0.0
    is_active: bool = False
    has_measurement: bool = False
    measured_this_frame: bool = False
    missed_frames: int = 0
    search_scale: float = 1.0
    velocity: Point = (0.0, 0.0)
    search_region: Optional[Rect] = None
    detection_area: float = 0.0
    threshold: float = 0.0


class CalibrationData:
    """Настройки обнаружения и сопровождения, включая обратную совместимость."""

    def __init__(self):
        # Начальные значения совпадают со встроенным пресетом «Базовый».
        self.min_area = 5
        self.max_area = 257
        self.circularity_threshold = 0.30
        self.brightness_threshold = 235
        self.contrast_threshold = 45
        self.mser_delta = 5  # Сохраняется для чтения старых JSON-файлов.
        self.blur_sigma = 0.49
        self.expected_reflectors = 2
        self.roi_region: Optional[Rect] = None

        # Одна стартовая область на один ожидаемый пик.
        self.peak_regions: List[Rect] = []

        # Новые настройки устойчивого интегрального пика.
        self.adaptive_threshold = False
        self.brightness_percentile = 98.5
        self.merge_radius = 3
        self.center_power = 3.50
        self.smoothing_alpha = 0.20
        self.roi_expand_step = 1
        self.roi_max_scale = 2.0
        self.lost_hold_frames = 75
        self.max_jump = 9.0

        # Индивидуальные параметры обнаружения: ключ — строковый номер Pn.
        # Отсутствие ключа означает использование глобальных параметров.
        self.region_settings: Dict[str, Dict] = {}

        # Независимые слои отображения.
        self.show_points = True
        self.show_circles = True
        self.show_frames = True
        self.show_lines = True
        self.show_distances = True
        self.show_distance_changes = True
        self.close_shape = True

        # Параметры одноразового поиска по отклику ИК-подсветки Hikvision.
        self.ir_settle_seconds = 2.0
        self.ir_flash_delta = 25
        self.ir_search_scale = 4.0
        self.ir_sample_count = 3
        self.hikvision_channel = 1
        self.ir_lock_enabled = True
        self.ir_lock_radius = 35.0
        self.ir_confirmed_centers: Dict[str, List[float]] = {}

    @staticmethod
    def _region(value) -> Optional[Rect]:
        if not value or len(value) != 4:
            return None
        return tuple(int(round(v)) for v in value)  # type: ignore[return-value]

    def to_dict(self) -> Dict:
        return {
            "min_area": self.min_area,
            "max_area": self.max_area,
            "circularity_threshold": self.circularity_threshold,
            "brightness_threshold": self.brightness_threshold,
            "contrast_threshold": self.contrast_threshold,
            "mser_delta": self.mser_delta,
            "blur_sigma": self.blur_sigma,
            "expected_reflectors": self.expected_reflectors,
            "roi_region": self.roi_region,
            "peak_regions": self.peak_regions,
            "adaptive_threshold": self.adaptive_threshold,
            "brightness_percentile": self.brightness_percentile,
            "merge_radius": self.merge_radius,
            "center_power": self.center_power,
            "smoothing_alpha": self.smoothing_alpha,
            "roi_expand_step": self.roi_expand_step,
            "roi_max_scale": self.roi_max_scale,
            "lost_hold_frames": self.lost_hold_frames,
            "max_jump": self.max_jump,
            "region_settings": self.region_settings,
            "show_points": self.show_points,
            "show_circles": self.show_circles,
            "show_frames": self.show_frames,
            "show_lines": self.show_lines,
            "show_distances": self.show_distances,
            "show_distance_changes": self.show_distance_changes,
            "close_shape": self.close_shape,
            "ir_settle_seconds": self.ir_settle_seconds,
            "ir_flash_delta": self.ir_flash_delta,
            "ir_search_scale": self.ir_search_scale,
            "ir_sample_count": self.ir_sample_count,
            "hikvision_channel": self.hikvision_channel,
            "ir_lock_enabled": self.ir_lock_enabled,
            "ir_lock_radius": self.ir_lock_radius,
            "ir_confirmed_centers": self.ir_confirmed_centers,
        }

    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4, ensure_ascii=False)

    def load(self, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        # get() позволяет загружать настройки от старой версии программы.
        for name in (
            "min_area",
            "max_area",
            "circularity_threshold",
            "brightness_threshold",
            "contrast_threshold",
            "mser_delta",
            "blur_sigma",
            "expected_reflectors",
            "adaptive_threshold",
            "brightness_percentile",
            "merge_radius",
            "center_power",
            "smoothing_alpha",
            "roi_expand_step",
            "roi_max_scale",
            "lost_hold_frames",
            "max_jump",
            "show_points",
            "show_circles",
            "show_frames",
            "show_lines",
            "show_distances",
            "show_distance_changes",
            "close_shape",
            "ir_settle_seconds",
            "ir_flash_delta",
            "ir_search_scale",
            "ir_sample_count",
            "hikvision_channel",
            "ir_lock_enabled",
            "ir_lock_radius",
        ):
            if name in data:
                setattr(self, name, data[name])

        self.roi_region = self._region(data.get("roi_region"))
        regions = []
        for region in data.get("peak_regions", []):
            parsed = self._region(region)
            if parsed is not None:
                regions.append(parsed)
        self.peak_regions = regions

        self.region_settings = {}
        raw_region_settings = data.get("region_settings", {})
        if isinstance(raw_region_settings, dict):
            for raw_id, raw_settings in raw_region_settings.items():
                try:
                    region_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if region_id < 1 or not isinstance(raw_settings, dict):
                    continue
                cleaned = {
                    name: raw_settings[name]
                    for name in REGION_DETECTION_FIELDS
                    if name in raw_settings
                }
                if cleaned:
                    self.region_settings[str(region_id)] = cleaned

        self.ir_confirmed_centers = {}
        raw_centers = data.get("ir_confirmed_centers", {})
        if isinstance(raw_centers, dict):
            for raw_id, raw_center in raw_centers.items():
                try:
                    region_id = int(raw_id)
                    if region_id < 1 or len(raw_center) != 2:
                        continue
                    center_x = float(raw_center[0])
                    center_y = float(raw_center[1])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(center_x) and np.isfinite(center_y):
                    self.ir_confirmed_centers[str(region_id)] = [center_x, center_y]


class HikvisionISAPI:
    """Минимальный клиент Hikvision ISAPI с Digest/Basic-аутентификацией."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        channel: int = 1,
        timeout: float = 5.0,
    ):
        parsed = urllib_parse.urlsplit(base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                "Адрес ISAPI должен иметь вид http://IP_КАМЕРЫ или https://IP_КАМЕРЫ"
            )
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        self.base_url = f"{parsed.scheme}://{host}{port}"
        self.channel = max(1, int(channel))
        self.timeout = max(1.0, float(timeout))

        password_manager = urllib_request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(
            None, self.base_url, username or "admin", password or ""
        )
        handlers = [
            urllib_request.HTTPDigestAuthHandler(password_manager),
            urllib_request.HTTPBasicAuthHandler(password_manager),
        ]
        if parsed.scheme == "https":
            # У камер часто установлен собственный сертификат. Соединение
            # остается внутри локальной сети и использует заданные реквизиты.
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib_request.HTTPSHandler(context=context))
        self.opener = urllib_request.build_opener(*handlers)
        self.ircut_endpoint: Optional[str] = None

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _find_xml_node(cls, root: ET.Element, local_name: str):
        for node in root.iter():
            if cls._local_name(node.tag).lower() == local_name.lower():
                return node
        return None

    def _request(
        self,
        path: str,
        method: str = "GET",
        body: Optional[bytes] = None,
    ) -> bytes:
        request = urllib_request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/xml",
                "Content-Type": "application/xml; charset=UTF-8",
                "User-Agent": "ReflectorTracker/5.6",
            },
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Hikvision отклонила авторизацию. Проверьте логин, пароль и "
                    "разрешение ISAPI/CGI в настройках камеры."
                ) from exc
            suffix = f": {detail[:180]}" if detail else ""
            raise RuntimeError(f"ISAPI HTTP {exc.code}{suffix}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Камера ISAPI недоступна: {exc.reason}") from exc

        if method == "PUT" and payload.strip():
            try:
                root = ET.fromstring(payload)
                status_code = self._find_xml_node(root, "statusCode")
                status_string = self._find_xml_node(root, "statusString")
                if status_code is not None and (status_code.text or "").strip() not in (
                    "1",
                    "OK",
                ):
                    detail = (
                        (status_string.text or "").strip()
                        if status_string is not None
                        else "неизвестная ошибка"
                    )
                    raise RuntimeError(f"Камера не применила режим: {detail}")
            except ET.ParseError:
                pass
        return payload

    def get_device_info(self) -> Dict[str, str]:
        payload = self._request("/ISAPI/System/deviceInfo")
        root = ET.fromstring(payload)
        result = {}
        for name in ("model", "deviceName", "firmwareVersion", "serialNumber"):
            node = self._find_xml_node(root, name)
            if node is not None and node.text:
                result[name] = node.text.strip()
        return result

    def get_ircut(self) -> Tuple[str, bytes, str]:
        paths = []
        if self.ircut_endpoint:
            paths.append(self.ircut_endpoint)
        paths.extend(
            [
                f"/ISAPI/Image/channels/{self.channel}/ircutFilter",
                f"/ISAPI/Image/channels/{self.channel}/IrcutFilter",
            ]
        )
        last_error = None
        for path in dict.fromkeys(paths):
            try:
                payload = self._request(path)
                root = ET.fromstring(payload)
                mode_node = self._find_xml_node(root, "IrcutFilterType")
                if mode_node is None:
                    raise RuntimeError("В ответе камеры нет IrcutFilterType")
                self.ircut_endpoint = path
                return (mode_node.text or "").strip().lower(), payload, path
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Камера не предоставила управление Day/Night через IrcutFilter. "
            f"Последняя ошибка: {last_error}"
        )

    def set_ircut(self, mode: str) -> str:
        mode = mode.strip().lower()
        if mode not in ("day", "night", "auto"):
            raise ValueError("Допустимые режимы Hikvision: day, night, auto")
        old_mode, payload, path = self.get_ircut()
        try:
            root = ET.fromstring(payload)
            mode_node = self._find_xml_node(root, "IrcutFilterType")
            if mode_node is None:
                raise ET.ParseError("IrcutFilterType missing")
            mode_node.text = mode
            if root.tag.startswith("{"):
                namespace = root.tag[1:].split("}", 1)[0]
                ET.register_namespace("", namespace)
            body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        except ET.ParseError:
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<IrcutFilter version="2.0" '
                'xmlns="http://www.hikvision.com/ver20/XMLSchema">'
                f"<IrcutFilterType>{mode}</IrcutFilterType>"
                "</IrcutFilter>"
            ).encode("utf-8")
        self._request(path, method="PUT", body=body)
        return old_mode


class RTSPCamera:
    """Неблокирующее получение последнего кадра RTSP через FFmpeg."""

    def __init__(self, url: str):
        self.url = url
        self.process: Optional[subprocess.Popen] = None
        self.is_connected = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.frame_size = 0
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_sequence = 0
        self.last_delivered_sequence = 0
        self.closed = False
        self.condition = threading.Condition()
        self.reader_thread: Optional[threading.Thread] = None

    def _probe_frame_size(self) -> Tuple[int, int]:
        ffprobe = require_executable("ffprobe")
        command = [
            ffprobe,
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            self.url,
        ]
        result = run_ffmpeg_command(command, timeout=10.0)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffprobe завершился с ошибкой")
        lines = result.stdout.strip().splitlines()
        dimensions = lines[0].strip() if lines else ""
        match = re.fullmatch(r"(\d+)x(\d+)", dimensions)
        if not match:
            raise RuntimeError(f"Неожиданный размер кадра: {dimensions}")
        return int(match.group(1)), int(match.group(2))

    def connect(self) -> bool:
        logger.info("Подключение к RTSP через FFmpeg")
        try:
            self.frame_width, self.frame_height = self._probe_frame_size()
            self.frame_size = self.frame_width * self.frame_height * 3
            ffmpeg = require_executable("ffmpeg")
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                self.url,
                "-an",
                "-sn",
                "-dn",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.error("Ошибка подключения к RTSP: %s", exc)
            return False

        self.closed = False
        self.latest_frame = None
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name="rtsp-frame-reader",
            daemon=True,
        )
        self.reader_thread.start()
        self.is_connected = True
        return True

    def _read_exactly(self, size: int) -> Optional[bytes]:
        if self.process is None or self.process.stdout is None:
            return None
        data = bytearray(size)
        view = memoryview(data)
        received = 0
        while received < size and not self.closed:
            chunk = self.process.stdout.read(size - received)
            if not chunk:
                return None
            view[received : received + len(chunk)] = chunk
            received += len(chunk)
        return bytes(data) if received == size else None

    def _reader_loop(self):
        while not self.closed:
            if self.process is None or self.process.poll() is not None:
                break
            raw_frame = self._read_exactly(self.frame_size)
            if raw_frame is None:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (self.frame_height, self.frame_width, 3)
            ).copy()
            with self.condition:
                self.latest_frame = frame
                self.latest_sequence += 1
                self.condition.notify_all()
        with self.condition:
            self.condition.notify_all()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected or self.closed:
            return False, None
        with self.condition:
            if (
                self.latest_frame is not None
                and self.latest_sequence != self.last_delivered_sequence
            ):
                self.last_delivered_sequence = self.latest_sequence
                self.frame_count += 1
                self.last_frame_time = time.time()
                return True, self.latest_frame.copy()
        return False, None

    def stop(self):
        self.is_connected = False
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        with self.condition:
            self.condition.notify_all()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=3)
        self.process = None


class WindowRecorder:
    """Фоновая запись RGB-кадров окна приложения в MP4 через FFmpeg."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self.is_recording = False
        self.output_path: Optional[str] = None
        self.width = 0
        self.height = 0
        self.fps = 5.0
        self.frames_written = 0
        self.frames_dropped = 0
        self.error_message = ""

    def start(self, output_path: str, width: int, height: int, fps: float):
        if self.is_recording:
            raise RuntimeError("Запись уже выполняется")
        ffmpeg = require_executable("ffmpeg")
        self.output_path = output_path
        self.width = max(2, int(width) - int(width) % 2)
        self.height = max(2, int(height) - int(height) % 2)
        self.fps = max(0.5, float(fps))
        self.frames_written = 0
        self.frames_dropped = 0
        self.error_message = ""
        self.frame_queue = queue.Queue(maxsize=3)

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.is_recording = True
        self.worker_thread = threading.Thread(
            target=self._worker,
            name="application-window-recorder",
            daemon=True,
        )
        self.worker_thread.start()

    def submit(self, rgb_frame: np.ndarray):
        if not self.is_recording:
            return
        if rgb_frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Неверный размер кадра записи: {rgb_frame.shape}; "
                f"ожидался {(self.height, self.width, 3)}"
            )
        frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frames_dropped += 1
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.frames_dropped += 1

    def _worker(self):
        try:
            while True:
                frame = self.frame_queue.get()
                if frame is None:
                    break
                if (
                    self.process is None
                    or self.process.stdin is None
                    or self.process.poll() is not None
                ):
                    raise RuntimeError("FFmpeg завершил работу во время записи")
                self.process.stdin.write(frame.tobytes())
                self.frames_written += 1
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            self.error_message = str(exc)
        finally:
            if self.process is not None:
                if self.process.stdin is not None:
                    try:
                        self.process.stdin.close()
                    except OSError:
                        pass
                try:
                    self.process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
            self.is_recording = False

    def stop(self):
        if self.worker_thread is None:
            self.is_recording = False
            return
        self.is_recording = False
        while True:
            try:
                self.frame_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    break
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=15)
        if self.worker_thread.is_alive() and self.process is not None:
            self.process.terminate()
            self.worker_thread.join(timeout=3)
        self.worker_thread = None
        self.process = None


class ReflectorDetector:
    """
    Выделяет один интегральный световой пик внутри одной области.

    Форма блика не является условием отбраковки. Сначала близкие светлые
    фрагменты объединяются, затем центр вычисляется по яркостным весам.
    """

    def __init__(self, calibration: CalibrationData):
        self.calibration = calibration
        self.background: Optional[np.ndarray] = None

    def _setting(self, settings: Optional[Dict], name: str):
        if settings is not None and name in settings:
            return settings[name]
        return getattr(self.calibration, name)

    def _signal(
        self,
        frame: np.ndarray,
        rect: Rect,
        settings: Optional[Dict] = None,
    ) -> np.ndarray:
        x, y, w, h = rect
        crop = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if self.background is not None and self.background.shape == frame.shape[:2]:
            background_crop = self.background[y : y + h, x : x + w]
            difference = cv2.absdiff(gray, background_crop.astype(np.uint8))
            # Фон помогает увидеть изменение, но исходная яркость остается,
            # поэтому неподвижный отражатель не исчезает.
            gray = np.maximum(gray, difference).astype(np.uint8)

        sigma = max(0.0, float(self._setting(settings, "blur_sigma")))
        if sigma > 0:
            kernel = min(31, max(3, int(round(sigma * 4)) | 1))
            gray = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        return gray

    @staticmethod
    def _border_level(signal: np.ndarray) -> float:
        if signal.shape[0] < 3 or signal.shape[1] < 3:
            return float(np.median(signal))
        border = np.concatenate(
            (signal[0, :], signal[-1, :], signal[1:-1, 0], signal[1:-1, -1])
        )
        return float(np.median(border))

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: Rect,
        predicted: Optional[Point] = None,
        detection_settings: Optional[Dict] = None,
        anchor: Optional[Point] = None,
        anchor_radius: Optional[float] = None,
    ) -> Optional[PeakDetection]:
        rect = clip_rect(region, frame.shape)
        if rect is None:
            return None
        x0, y0, width, height = rect
        signal = self._signal(frame, rect, detection_settings)
        signal_float = signal.astype(np.float32)
        background_level = self._border_level(signal)
        peak_value = float(np.max(signal_float))
        local_contrast = peak_value - background_level

        brightness = float(
            self._setting(detection_settings, "brightness_threshold")
        )
        contrast = max(
            1.0, float(self._setting(detection_settings, "contrast_threshold"))
        )
        adaptive = bool(
            self._setting(detection_settings, "adaptive_threshold")
        )

        absolute_ok = peak_value >= brightness
        adaptive_ok = (
            adaptive
            and peak_value >= max(20.0, brightness * 0.65)
            and local_contrast >= contrast * 1.5
        )
        if not (absolute_ok or adaptive_ok) or local_contrast < contrast:
            return None

        if adaptive:
            percentile = float(
                np.percentile(
                    signal_float,
                    np.clip(
                        self._setting(
                            detection_settings, "brightness_percentile"
                        ),
                        80.0,
                        99.95,
                    ),
                )
            )
            shape_level = background_level + 0.38 * local_contrast
            allowed_floor = brightness if absolute_ok else brightness * 0.65
            threshold = max(
                allowed_floor,
                background_level + 0.5 * contrast,
                min(percentile, shape_level),
            )
        else:
            threshold = brightness
        threshold = min(threshold, peak_value - 1.0)

        binary = np.where(signal_float >= threshold, 255, 0).astype(np.uint8)
        merge_radius = max(
            0,
            int(round(self._setting(detection_settings, "merge_radius"))),
        )
        if merge_radius > 0:
            kernel_size = min(61, merge_radius * 2 + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if count <= 1:
            return None

        min_area = max(
            1, int(round(self._setting(detection_settings, "min_area")))
        )
        max_area = max(
            min_area,
            int(round(self._setting(detection_settings, "max_area"))),
        )
        candidates = []
        positive = np.maximum(signal_float - background_level, 0.0)
        power = float(
            np.clip(
                self._setting(detection_settings, "center_power"), 1.0, 5.0
            )
        )
        powered = np.power(positive, power)
        predicted_local = None
        if predicted is not None:
            predicted_local = np.array([predicted[0] - x0, predicted[1] - y0])
        anchor_local = None
        if anchor is not None:
            anchor_local = np.array([anchor[0] - x0, anchor[1] - y0])
        allowed_anchor_radius = max(3.0, float(anchor_radius or 0.0))
        distance_scale = max(10.0, 0.35 * np.hypot(width, height))

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            component_mask = labels == label
            energy = float(np.sum(powered[component_mask]))
            center = centroids[label]
            proximity = 1.0
            if predicted_local is not None:
                distance = float(np.linalg.norm(center - predicted_local))
                proximity = float(np.exp(-((distance / distance_scale) ** 2)))
            anchor_proximity = 1.0
            if anchor_local is not None:
                anchor_distance = float(np.linalg.norm(center - anchor_local))
                if anchor_distance > allowed_anchor_radius:
                    continue
                anchor_scale = max(4.0, allowed_anchor_radius * 0.45)
                anchor_proximity = float(
                    np.exp(-((anchor_distance / anchor_scale) ** 2))
                )
            # Без ИК-якоря сохраняется прежнее поведение. При наличии якоря
            # посторонний большой белый объект больше не может победить только
            # за счет площади и энергии.
            predicted_weight = 0.30 + 0.70 * proximity
            anchor_weight = (
                0.015 + 0.985 * anchor_proximity
                if anchor_local is not None
                else 1.0
            )
            score = energy * predicted_weight * anchor_weight
            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            combined_proximity = float(np.sqrt(proximity * anchor_proximity))
            candidates.append((score, label, bbox, combined_proximity))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_label, best_bbox, best_proximity = candidates[0]

        # Дополнительно присоединяем соседние компоненты. Это важно, когда
        # пересвеченная призма содержит несколько разорванных ярких участков.
        selected_labels = {best_label}
        selected_boxes = [best_bbox]
        join_distance = max(1.0, merge_radius * 1.5)
        changed = True
        while changed:
            changed = False
            for _, label, bbox, _ in candidates:
                if label in selected_labels:
                    continue
                if any(rect_distance(bbox, box) <= join_distance for box in selected_boxes):
                    selected_labels.add(label)
                    selected_boxes.append(bbox)
                    changed = True

        cluster = np.isin(labels, list(selected_labels)).astype(np.uint8) * 255

        # Берем не только насыщенное ядро, но и световой ореол вокруг него.
        halo_radius = max(1, merge_radius // 2)
        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (halo_radius * 2 + 1, halo_radius * 2 + 1)
        )
        neighborhood = cv2.dilate(cluster, halo_kernel)
        halo_threshold = max(
            background_level + 0.18 * local_contrast,
            min(threshold, brightness * 0.65),
        )
        final_mask = (neighborhood > 0) & (signal_float >= halo_threshold)
        final_area = int(np.count_nonzero(final_mask))
        if final_area < min_area:
            return None

        weights = powered * final_mask.astype(np.float32)
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            return None
        yy, xx = np.indices(signal.shape, dtype=np.float32)
        center_x = float(np.sum(xx * weights) / total_weight + x0)
        center_y = float(np.sum(yy * weights) / total_weight + y0)

        mask_u8 = final_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        circularity = 0.0
        if contours:
            merged_contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(merged_contour))
            perimeter = float(cv2.arcLength(merged_contour, True))
            if perimeter > 0:
                circularity = float(4.0 * np.pi * contour_area / (perimeter**2))

        ys, xs = np.where(final_mask)
        bbox_global = (
            int(xs.min() + x0),
            int(ys.min() + y0),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        radius = float(np.sqrt(final_area / np.pi))

        contrast_score = float(np.clip(local_contrast / (contrast * 3.0), 0.0, 1.0))
        brightness_score = float(np.clip(peak_value / max(brightness, 1.0), 0.0, 1.0))
        area_score = float(np.clip(final_area / max(min_area * 2.0, 1.0), 0.0, 1.0))
        expected_circularity = max(
            0.05,
            float(
                self._setting(detection_settings, "circularity_threshold")
            ),
        )
        shape_score = float(np.clip(circularity / expected_circularity, 0.0, 1.0))
        confidence = float(
            np.clip(
                0.35 * contrast_score
                + 0.25 * brightness_score
                + 0.20 * area_score
                + 0.10 * best_proximity
                + 0.10 * shape_score,
                0.0,
                1.0,
            )
        )

        return PeakDetection(
            position=(center_x, center_y),
            radius=radius,
            confidence=confidence,
            area=float(final_area),
            circularity=circularity,
            bbox=bbox_global,
            threshold=threshold,
            max_intensity=peak_value,
        )

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sigma = max(0.0, float(self.calibration.blur_sigma))
        if sigma > 0:
            kernel = min(31, max(3, int(round(sigma * 4)) | 1))
            gray = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
        _, thresholded = cv2.threshold(
            gray,
            int(round(self.calibration.brightness_threshold)),
            255,
            cv2.THRESH_TOZERO,
        )
        if self.calibration.peak_regions:
            mask = np.zeros_like(thresholded)
            for region in self.calibration.peak_regions:
                clipped = clip_rect(region, frame.shape)
                if clipped is None:
                    continue
                x, y, w, h = clipped
                mask[y : y + h, x : x + w] = 255
            thresholded = cv2.bitwise_and(thresholded, mask)
        return thresholded


class ReflectorTracker:
    """Один постоянный трек на каждую назначенную пользователем область."""

    def __init__(self, calibration: CalibrationData):
        self.calibration = calibration
        self.detector = ReflectorDetector(calibration)
        self.tracks: List[ReflectorPoint] = []
        self.position_history: Dict[int, deque] = {}
        self.history_length = 60
        self.last_report_time = time.time()
        self.report_interval = 10.0
        self._region_signature = ()
        self.previous_edge_lengths: Dict[Tuple[int, int], float] = {}
        self.reset_regions()

    def reset_regions(self):
        self.tracks.clear()
        self.position_history.clear()
        self.previous_edge_lengths.clear()
        valid_regions = []
        for index, region in enumerate(self.calibration.peak_regions):
            x, y, w, h = [int(v) for v in region]
            if w < 3 or h < 3:
                continue
            center = (x + w / 2.0, y + h / 2.0)
            self.tracks.append(
                ReflectorPoint(id=index + 1, base_region=(x, y, w, h), position=center)
            )
            self.position_history[index + 1] = deque(maxlen=self.history_length)
            valid_regions.append((x, y, w, h))
        self._region_signature = tuple(valid_regions)

    def update_settings(self):
        current_signature = tuple(tuple(int(v) for v in r) for r in self.calibration.peak_regions)
        if current_signature != self._region_signature:
            background = self.detector.background
            self.reset_regions()
            self.detector.background = background

    @staticmethod
    def _centered_region(
        center: Point,
        base_region: Rect,
        scale: float,
        frame_shape: Tuple[int, ...],
    ) -> Rect:
        _, _, base_w, base_h = base_region
        width = max(3, int(round(base_w * scale)))
        height = max(3, int(round(base_h * scale)))
        candidate = (
            int(round(center[0] - width / 2.0)),
            int(round(center[1] - height / 2.0)),
            width,
            height,
        )
        clipped = clip_rect(candidate, frame_shape)
        return clipped if clipped is not None else base_region

    def _update_track(self, track: ReflectorPoint, frame: np.ndarray, now: float):
        base_x, base_y, base_w, base_h = track.base_region
        base_center = (base_x + base_w / 2.0, base_y + base_h / 2.0)
        predicted = (
            track.position[0] + track.velocity[0],
            track.position[1] + track.velocity[1],
        )
        if not track.has_measurement:
            predicted = base_center

        # Область расширяется вокруг исходного положения, выбранного
        # пользователем, и больше не «шагает» вслед за случайным бликом.
        search_region = self._centered_region(
            base_center,
            track.base_region,
            track.search_scale,
            frame.shape,
        )
        detection_settings = self.calibration.region_settings.get(str(track.id))
        ir_anchor = None
        ir_anchor_radius = None
        if self.calibration.ir_lock_enabled:
            saved_anchor = self.calibration.ir_confirmed_centers.get(str(track.id))
            if saved_anchor and len(saved_anchor) == 2:
                ir_anchor = (float(saved_anchor[0]), float(saved_anchor[1]))
                ir_anchor_radius = max(
                    float(self.calibration.ir_lock_radius),
                    float(self.calibration.max_jump) * 2.0,
                )
        detection = self.detector.detect_in_region(
            frame,
            search_region,
            predicted,
            detection_settings=detection_settings,
            anchor=ir_anchor,
            anchor_radius=ir_anchor_radius,
        )

        # В обычном режиме действует строгий лимит скачка. После потери поиск
        # разрешается во всей текущей (но неподвижной) области, при этом новый
        # блик должен быть похож по площади на ранее отслеживаемый.
        if detection is not None and track.has_measurement:
            reference = track.position if track.missed_frames > 0 else predicted
            jump = float(
                np.linalg.norm(np.array(detection.position) - np.array(reference))
            )
            allowed_jump = float(self.calibration.max_jump)
            if track.missed_frames > 0:
                allowed_jump = max(
                    allowed_jump,
                    0.5 * float(np.hypot(search_region[2], search_region[3])),
                )
                if track.detection_area > 0:
                    area_ratio = detection.area / track.detection_area
                    if area_ratio < 0.40 or area_ratio > 2.50:
                        detection = None
            if detection is not None and jump > allowed_jump:
                detection = None

        if detection is not None:
            measurement = np.array(detection.position, dtype=np.float64)
            old_position = np.array(track.position, dtype=np.float64)
            old_velocity = np.array(track.velocity, dtype=np.float64)
            if (
                not track.has_measurement
                or not track.is_active
                or track.missed_frames > 5
            ):
                filtered = measurement
                velocity = np.zeros(2, dtype=np.float64)
            else:
                alpha = float(np.clip(self.calibration.smoothing_alpha, 0.02, 1.0))
                prediction = old_position + old_velocity
                filtered = (1.0 - alpha) * prediction + alpha * measurement
                instantaneous = filtered - old_position
                velocity = 0.70 * old_velocity + 0.30 * instantaneous

            track.position = (float(filtered[0]), float(filtered[1]))
            track.velocity = (float(velocity[0]), float(velocity[1]))
            track.radius = 0.65 * track.radius + 0.35 * detection.radius
            track.confidence = detection.confidence
            track.last_seen = now
            track.is_active = True
            track.has_measurement = True
            track.measured_this_frame = True
            track.missed_frames = 0
            track.detection_area = detection.area
            track.threshold = detection.threshold
            shrinking_scale = max(
                1.0,
                track.search_scale - max(0.12, (track.search_scale - 1.0) * 0.35),
            )

            # Если отражатель найден за исходной границей, не сжимаем область
            # настолько, чтобы он снова оказался снаружи на следующем кадре.
            margin = max(4.0, track.radius + 3.0)
            required_scale = max(
                1.0,
                2.0 * (abs(track.position[0] - base_center[0]) + margin) / base_w,
                2.0 * (abs(track.position[1] - base_center[1]) + margin) / base_h,
            )
            track.search_scale = min(
                max(1.0, float(self.calibration.roi_max_scale)),
                max(shrinking_scale, required_scale),
            )
            self.position_history[track.id].append(track.position)
        else:
            track.measured_this_frame = False
            track.missed_frames += 1
            track.velocity = (track.velocity[0] * 0.85, track.velocity[1] * 0.85)
            track.confidence *= 0.985
            track.is_active = (
                track.has_measurement
                and track.missed_frames <= int(self.calibration.lost_hold_frames)
            )
            base_min_size = max(3.0, float(min(track.base_region[2], track.base_region[3])))
            scale_step = 2.0 * float(self.calibration.roi_expand_step) / base_min_size
            track.search_scale = min(
                max(1.0, float(self.calibration.roi_max_scale)),
                track.search_scale + scale_step,
            )

        track.search_region = self._centered_region(
            base_center,
            track.base_region,
            track.search_scale,
            frame.shape,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.update_settings()
        now = time.time()
        for track in self.tracks:
            self._update_track(track, frame, now)

        quality = self._evaluate_quality()
        display = self._visualize(frame, quality)
        if now - self.last_report_time >= self.report_interval:
            self._print_report()
            self.last_report_time = now
        return display, quality

    def _evaluate_quality(self) -> Dict:
        active = [track for track in self.tracks if track.is_active]
        measured = [track for track in active if track.measured_this_frame]
        held = [track for track in active if not track.measured_this_frame]
        stability = float(np.mean([t.confidence for t in active])) if active else 0.0
        expected = max(1, int(self.calibration.expected_reflectors))
        return {
            "active_tracks": len(active),
            "measured_tracks": len(measured),
            "held_tracks": len(held),
            "expected_tracks": expected,
            "coverage_ratio": len(active) / expected,
            "tracking_stability": stability,
            "timestamp": time.time(),
        }

    def _visualize(self, frame: np.ndarray, quality: Dict) -> np.ndarray:
        display = frame.copy()
        if not self.tracks:
            cv2.putText(
                display,
                "Select one search region for each peak",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )
            return display

        calibration = self.calibration

        def outlined_text(text, origin, scale, color, thickness=1):
            cv2.putText(
                display,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        for track in self.tracks:
            bx, by, bw, bh = track.base_region
            uses_individual = str(track.id) in calibration.region_settings
            uses_ir_anchor = (
                calibration.ir_lock_enabled
                and str(track.id) in calibration.ir_confirmed_centers
            )

            if calibration.show_frames:
                cv2.rectangle(
                    display,
                    (bx, by),
                    (bx + bw, by + bh),
                    (160, 160, 160),
                    1,
                )

            if calibration.show_frames and track.search_region is not None:
                sx, sy, sw, sh = track.search_region
                if track.measured_this_frame:
                    region_color = (0, 220, 0)
                elif track.is_active:
                    region_color = (0, 210, 255)
                else:
                    region_color = (0, 0, 255)
                cv2.rectangle(
                    display, (sx, sy), (sx + sw, sy + sh), region_color, 1
                )
                frame_label = f"P{track.id}"
                if uses_individual:
                    frame_label += " IND"
                if uses_ir_anchor:
                    frame_label += " IR"
                outlined_text(
                    frame_label,
                    (sx, max(15, sy - 5)),
                    0.42,
                    region_color,
                    1,
                )

            if track.has_measurement:
                px, py = int(round(track.position[0])), int(round(track.position[1]))
                if track.measured_this_frame:
                    point_color = (0, 255, 0)
                    state = "LOCK"
                elif track.is_active:
                    point_color = (0, 190, 255)
                    state = f"HOLD {track.missed_frames}"
                else:
                    point_color = (0, 0, 255)
                    state = "LOST"
                if calibration.show_circles:
                    cv2.circle(
                        display,
                        (px, py),
                        max(5, int(round(track.radius))),
                        point_color,
                        2,
                        cv2.LINE_AA,
                    )
                if calibration.show_points:
                    cv2.circle(display, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    outlined_text(
                        f"P{track.id} {state} C:{track.confidence:.2f}",
                        (px + 9, py - 9),
                        0.48,
                        (255, 255, 255),
                        1,
                    )
            elif calibration.show_points and not calibration.show_frames:
                outlined_text(
                    f"P{track.id} SEARCH",
                    (bx, max(15, by - 5)),
                    0.45,
                    (0, 0, 255),
                    1,
                )

        # Топология сохраняет порядок P1-P2-P3 даже при потере одной точки:
        # отсутствующую вершину нельзя незаметно заменить диагональю P1-P3.
        ordered_tracks = sorted(self.tracks, key=lambda track: track.id)
        visible = {
            track.id: track
            for track in ordered_tracks
            if track.has_measurement and track.is_active
        }
        edges = []
        for first, second in zip(ordered_tracks, ordered_tracks[1:]):
            if first.id in visible and second.id in visible:
                edges.append((visible[first.id], visible[second.id]))
        all_vertices_visible = len(visible) == len(ordered_tracks)
        if (
            calibration.close_shape
            and len(ordered_tracks) >= 3
            and all_vertices_visible
        ):
            edges.append((visible[ordered_tracks[-1].id], visible[ordered_tracks[0].id]))

        current_edge_lengths: Dict[Tuple[int, int], float] = {}
        for edge_index, (first, second) in enumerate(edges):
            first_point = (
                int(round(first.position[0])),
                int(round(first.position[1])),
            )
            second_point = (
                int(round(second.position[0])),
                int(round(second.position[1])),
            )
            length = float(
                np.linalg.norm(
                    np.asarray(second.position, dtype=np.float64)
                    - np.asarray(first.position, dtype=np.float64)
                )
            )
            edge_key = (first.id, second.id)
            current_edge_lengths[edge_key] = length
            previous = self.previous_edge_lengths.get(edge_key)

            if calibration.show_lines:
                cv2.line(
                    display,
                    first_point,
                    second_point,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

            if calibration.show_distances or calibration.show_distance_changes:
                middle_x = int(round((first_point[0] + second_point[0]) / 2.0))
                middle_y = int(round((first_point[1] + second_point[1]) / 2.0))
                dx = second_point[0] - first_point[0]
                dy = second_point[1] - first_point[1]
                norm = max(1.0, float(np.hypot(dx, dy)))
                normal_x = int(round(-dy / norm * 12.0))
                normal_y = int(round(dx / norm * 12.0))
                label_x = middle_x + normal_x
                label_y = middle_y + normal_y
                if edge_index % 2:
                    label_x -= 2 * normal_x
                    label_y -= 2 * normal_y
                label_x = max(5, min(max(5, display.shape[1] - 175), label_x))
                bottom_margin = 30 if (
                    calibration.show_distances
                    and calibration.show_distance_changes
                ) else 10
                label_y = max(
                    15,
                    min(max(15, display.shape[0] - bottom_margin), label_y),
                )

                if calibration.show_distances:
                    outlined_text(
                        f"P{first.id}-P{second.id}: {length:.2f} px",
                        (label_x, label_y),
                        0.43,
                        (255, 255, 255),
                        1,
                    )
                    label_y += 17
                if calibration.show_distance_changes:
                    change_text = (
                        f"dL: {length - previous:+.2f} px"
                        if previous is not None
                        else "dL: --"
                    )
                    outlined_text(
                        change_text,
                        (label_x, label_y),
                        0.40,
                        (0, 230, 255),
                        1,
                    )

        # Сравнение следующего кадра всегда идет с текущей геометрией,
        # независимо от того, включен ли визуальный слой изменений.
        self.previous_edge_lengths = current_edge_lengths

        color = (0, 255, 0) if quality["coverage_ratio"] >= 0.99 else (0, 210, 255)
        cv2.putText(
            display,
            (
                f"Peaks: {quality['active_tracks']}/{quality['expected_tracks']}  "
                f"measured:{quality['measured_tracks']} held:{quality['held_tracks']}"
            ),
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
        return display

    def _print_report(self):
        active = [track for track in self.tracks if track.is_active]
        print(f"\n{'=' * 60}")
        print(f"Отчет [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"Активных пиков: {len(active)}/{self.calibration.expected_reflectors}")
        for track in sorted(active, key=lambda item: item.id):
            state = "измерен" if track.measured_this_frame else "удерживается"
            print(
                f"P{track.id}: x={track.position[0]:.3f}, y={track.position[1]:.3f}, "
                f"C={track.confidence:.2f}, {state}"
            )
        print(f"{'=' * 60}\n")


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9),
            wraplength=340,
        )
        label.pack(ipadx=2, ipady=2)

    def hide_tip(self, event=None):
        window = self.tip_window
        self.tip_window = None
        if window:
            window.destroy()


class ReflectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Система устойчивого отслеживания отражателей v5.6")
        self.root.geometry("1500x940")
        self.root.minsize(1050, 680)

        self.is_running = False
        self.is_calibrating = False
        self.show_preprocessing = False
        self.use_rtsp = True
        self.rtsp_camera: Optional[RTSPCamera] = None
        self.local_cap = None
        self.current_frame: Optional[np.ndarray] = None
        self.background_frame: Optional[np.ndarray] = None
        self.tracker: Optional[ReflectorTracker] = None
        self.calibration = CalibrationData()
        self._last_frame_time = time.time()
        self._widgets_ready = False
        self._settings_reset_job = None
        self.region_settings_window = None
        self.region_dialog_selector_var = None
        self.region_dialog_enabled_var = None
        self.region_dialog_vars: Dict[str, tk.Variable] = {}
        self.region_dialog_edit_widgets = []
        self.hikvision_control: Optional[HikvisionISAPI] = None
        self._ir_scan_token = 0
        self._ir_scan_running = False
        self._ir_day_samples: List[np.ndarray] = []
        self._ir_night_samples: List[np.ndarray] = []
        self._ir_original_mode = "auto"
        self._ir_preview_overlay = None
        self._ir_preview_until = 0.0

        # Масштаб относится только к просмотру. Обработка всегда выполняется
        # по исходному кадру полного разрешения.
        self.view_zoom = 1.0
        self.view_center = (0.5, 0.5)
        self._pan_start = None
        self._view_render_size = (1, 1)

        self.current_preset_name = "Базовый"
        self.user_presets: Dict[str, Dict] = {}
        self.user_preset_store_path = self._get_user_preset_store_path()
        self._load_user_preset_store()
        self.window_recorder = WindowRecorder()
        self._record_capture_job = None
        self._record_log_file = None
        self._record_log_writer = None
        self._record_start_time = 0.0
        self._record_frame_index = 0
        self._record_output_path: Optional[str] = None

        self._create_widgets()
        self._widgets_ready = True
        self._update_calibration_from_ui()
        self._create_menu()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._update_display()

    def _scroll_controls(self, event):
        if event.delta:
            self.control_canvas.yview_scroll(int(-event.delta / 120), "units")

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _clamp_view_center(self):
        half = 0.5 / max(1.0, self.view_zoom)
        self.view_center = (
            self._clamp(self.view_center[0], half, 1.0 - half),
            self._clamp(self.view_center[1], half, 1.0 - half),
        )

    def _set_view_zoom(self, new_zoom: float, event=None):
        old_zoom = self.view_zoom
        new_zoom = self._clamp(float(new_zoom), 1.0, 12.0)
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        # При масштабировании колесом сохраняем под курсором ту же точку кадра.
        if event is not None and old_zoom >= 1.0:
            canvas_w = max(1, self.main_canvas.winfo_width())
            canvas_h = max(1, self.main_canvas.winfo_height())
            render_w, render_h = self._view_render_size
            render_w = max(1, render_w)
            render_h = max(1, render_h)
            image_left = (canvas_w - render_w) / 2.0
            image_top = (canvas_h - render_h) / 2.0
            fraction_x = self._clamp((event.x - image_left) / render_w, 0.0, 1.0)
            fraction_y = self._clamp((event.y - image_top) / render_h, 0.0, 1.0)

            old_left = self.view_center[0] - 0.5 / old_zoom
            old_top = self.view_center[1] - 0.5 / old_zoom
            image_x = old_left + fraction_x / old_zoom
            image_y = old_top + fraction_y / old_zoom
            self.view_center = (
                image_x - (fraction_x - 0.5) / new_zoom,
                image_y - (fraction_y - 0.5) / new_zoom,
            )

        self.view_zoom = new_zoom
        if self.view_zoom <= 1.0001:
            self.view_center = (0.5, 0.5)
        self._clamp_view_center()
        if hasattr(self, "zoom_label"):
            self.zoom_label.config(text=f"{self.view_zoom:.2f}×")
        if hasattr(self, "main_canvas"):
            self.main_canvas.config(cursor="fleur" if self.view_zoom > 1.0 else "")

    def _zoom_wheel(self, event):
        direction = 0
        if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4:
            direction = 1
        elif getattr(event, "delta", 0) < 0 or getattr(event, "num", None) == 5:
            direction = -1
        if direction > 0:
            self._set_view_zoom(self.view_zoom * 1.25, event)
        elif direction < 0:
            self._set_view_zoom(self.view_zoom / 1.25, event)
        return "break"

    def _zoom_in(self):
        self._set_view_zoom(self.view_zoom * 1.25)

    def _zoom_out(self):
        self._set_view_zoom(self.view_zoom / 1.25)

    def reset_view_zoom(self):
        self.view_center = (0.5, 0.5)
        self.view_zoom = 1.0
        if hasattr(self, "zoom_label"):
            self.zoom_label.config(text="1.00×")
        if hasattr(self, "main_canvas"):
            self.main_canvas.config(cursor="")

    def _start_pan(self, event):
        if self.view_zoom <= 1.0:
            self._pan_start = None
            return
        self._pan_start = (event.x, event.y, self.view_center)

    def _pan_view(self, event):
        if self._pan_start is None or self.view_zoom <= 1.0:
            return
        start_x, start_y, start_center = self._pan_start
        render_w, render_h = self._view_render_size
        delta_x = (event.x - start_x) / max(1, render_w) / self.view_zoom
        delta_y = (event.y - start_y) / max(1, render_h) / self.view_zoom
        self.view_center = (
            start_center[0] - delta_x,
            start_center[1] - delta_y,
        )
        self._clamp_view_center()

    def _end_pan(self, event=None):
        self._pan_start = None

    def _schedule_processing_reset(self):
        """Сбрасывает трекер один раз после окончания движения ползунка."""
        if not self._widgets_ready:
            return
        if self._settings_reset_job is not None:
            try:
                self.root.after_cancel(self._settings_reset_job)
            except tk.TclError:
                pass
        self._settings_reset_job = self.root.after(
            180, self._reset_processing_modules
        )

    def _reset_processing_modules(self, announce: bool = True):
        self._settings_reset_job = None
        if self.tracker is None:
            return
        self.tracker = ReflectorTracker(self.calibration)
        self.tracker.detector.background = self.background_frame
        if announce:
            self.status_label.config(
                text=(
                    "Параметры изменены: детектор и трекер полностью сброшены; "
                    "поиск начат заново."
                )
            )

    @staticmethod
    def _get_user_preset_store_path() -> Path:
        app_data = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if app_data:
            return Path(app_data) / "ReflectorTracker" / "user_presets.json"
        return Path.home() / ".reflector_tracker" / "user_presets.json"

    def _load_user_preset_store(self):
        self.user_presets = {}
        path = self.user_preset_store_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            raw_presets = data.get("presets", data)
            if not isinstance(raw_presets, dict):
                raise ValueError("Неверный формат хранилища пользовательских пресетов")
            for raw_name, parameters in raw_presets.items():
                name = str(raw_name).strip()
                if (
                    name
                    and name not in BUILTIN_PRESETS
                    and isinstance(parameters, dict)
                ):
                    self.user_presets[name] = parameters
        except Exception as exc:
            logger.error("Не удалось загрузить пользовательские пресеты: %s", exc)

    def _save_user_preset_store(self):
        path = self.user_preset_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "reflector-tracker-user-presets",
            "version": 1,
            "presets": self.user_presets,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def _all_preset_names(self) -> List[str]:
        return list(BUILTIN_PRESETS.keys()) + sorted(
            self.user_presets.keys(), key=str.casefold
        )

    def _refresh_preset_combo(self, selected: Optional[str] = None):
        if not hasattr(self, "preset_combo"):
            return
        names = self._all_preset_names()
        self.preset_combo.config(values=names)
        if selected and selected in names:
            self.preset_var.set(selected)
        elif self.preset_var.get() not in names:
            self.preset_var.set("Базовый")

    def _add_scale(self, parent, label, variable, start, end, digits=0, tooltip=""):
        header = ttk.Frame(parent)
        header.pack(fill=tk.X)
        ttk.Label(header, text=label).pack(side=tk.LEFT)
        shown_value = tk.StringVar()
        value_label = ttk.Label(
            header, width=7, anchor=tk.E, textvariable=shown_value
        )
        value_label.pack(side=tk.RIGHT)

        def refresh_label(*_):
            value = variable.get()
            shown_value.set(f"{value:.{digits}f}")

        def changed(_=None):
            if self._widgets_ready:
                self._on_param_change()

        scale = ttk.Scale(parent, from_=start, to=end, variable=variable, command=changed)
        scale.pack(fill=tk.X, pady=(0, 3))
        variable.trace_add("write", refresh_label)
        refresh_label()
        if tooltip:
            ToolTip(scale, tooltip)
        return scale

    def _create_widgets(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_holder = ttk.Frame(main, width=370)
        control_holder.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_holder.pack_propagate(False)
        self.control_canvas = tk.Canvas(control_holder, highlightthickness=0, width=345)
        scrollbar = ttk.Scrollbar(
            control_holder, orient=tk.VERTICAL, command=self.control_canvas.yview
        )
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(self.control_canvas)
        controls_window = self.control_canvas.create_window(
            (0, 0), window=controls, anchor=tk.NW
        )
        controls.bind(
            "<Configure>",
            lambda event: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            ),
        )
        self.control_canvas.bind(
            "<Configure>",
            lambda event: self.control_canvas.itemconfigure(
                controls_window, width=event.width
            ),
        )
        self.control_canvas.bind("<Enter>", lambda event: self.root.bind_all("<MouseWheel>", self._scroll_controls))
        self.control_canvas.bind("<Leave>", lambda event: self.root.unbind_all("<MouseWheel>"))

        source = ttk.LabelFrame(controls, text="Источник видео")
        source.pack(fill=tk.X, pady=4)
        self.source_type_var = tk.StringVar(value="rtsp")
        ttk.Radiobutton(
            source,
            text="RTSP-поток (IP-камера)",
            variable=self.source_type_var,
            value="rtsp",
            command=self._on_source_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            source,
            text="Локальная камера",
            variable=self.source_type_var,
            value="local",
            command=self._on_source_change,
        ).pack(anchor=tk.W)
        self.rtsp_frame = ttk.Frame(source)
        self.rtsp_frame.pack(fill=tk.X, pady=3)
        ttk.Label(self.rtsp_frame, text="RTSP URL:").pack(anchor=tk.W)
        self.rtsp_url_var = tk.StringVar(
            value="rtsp://admin:ParallelogramM1!@10.242.52.118:554/Streaming/Channels/101"
        )
        ttk.Entry(self.rtsp_frame, textvariable=self.rtsp_url_var).pack(fill=tk.X)
        self.local_frame = ttk.Frame(source)
        ttk.Label(self.local_frame, text="ID камеры:").pack(side=tk.LEFT)
        self.local_camera_var = tk.StringVar(value="0")
        ttk.Combobox(
            self.local_frame,
            textvariable=self.local_camera_var,
            values=["0", "1", "2", "3"],
            width=6,
        ).pack(side=tk.LEFT, padx=4)
        self.connection_status = ttk.Label(
            source, text="Не подключено", foreground="red"
        )
        self.connection_status.pack(anchor=tk.W, pady=3)

        actions = ttk.LabelFrame(controls, text="Управление")
        actions.pack(fill=tk.X, pady=4)
        self.start_btn = ttk.Button(
            actions, text="▶ Запуск отслеживания", command=self.toggle_detection
        )
        self.start_btn.pack(fill=tk.X, pady=2)
        self.calibrate_btn = ttk.Button(
            actions, text="⚙ Режим калибровки", command=self.toggle_calibration
        )
        self.calibrate_btn.pack(fill=tk.X, pady=2)
        ttk.Button(
            actions,
            text="🎯 Задать области всех пиков",
            command=self.select_peak_regions,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            actions, text="Очистить области пиков", command=self.clear_peak_regions
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            actions, text="📸 Захватить фон", command=self.capture_background
        ).pack(fill=tk.X, pady=2)

        zoom_row = ttk.Frame(actions)
        zoom_row.pack(fill=tk.X, pady=3)
        ttk.Label(zoom_row, text="Просмотр:").pack(side=tk.LEFT)
        ttk.Button(zoom_row, text="−", width=3, command=self._zoom_out).pack(
            side=tk.LEFT, padx=(5, 1)
        )
        self.zoom_label = ttk.Label(zoom_row, text="1.00×", width=7, anchor=tk.CENTER)
        self.zoom_label.pack(side=tk.LEFT)
        ttk.Button(zoom_row, text="+", width=3, command=self._zoom_in).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(
            zoom_row, text="Сброс", width=7, command=self.reset_view_zoom
        ).pack(side=tk.RIGHT)
        ToolTip(
            zoom_row,
            "Колесо мыши над изображением — приблизить/отдалить. "
            "Левая кнопка с перетаскиванием — переместить увеличенный кадр.",
        )
        self.region_status_label = ttk.Label(actions, text="Области: 0/4")
        self.region_status_label.pack(anchor=tk.W, pady=2)

        recording = ttk.LabelFrame(controls, text="Запись наблюдения")
        recording.pack(fill=tk.X, pady=4)
        recording_row = ttk.Frame(recording)
        recording_row.pack(fill=tk.X, pady=2)
        ttk.Label(recording_row, text="Кадров в секунду:").pack(side=tk.LEFT)
        self.record_fps_var = tk.StringVar(value="5")
        ttk.Combobox(
            recording_row,
            textvariable=self.record_fps_var,
            values=["1", "2", "5", "10", "15", "25"],
            width=6,
            state="readonly",
        ).pack(side=tk.RIGHT)
        self.record_btn = ttk.Button(
            recording,
            text="⏺ Начать запись окна",
            command=self.toggle_window_recording,
        )
        self.record_btn.pack(fill=tk.X, pady=2)
        self.record_status_label = ttk.Label(
            recording, text="Запись выключена", foreground="gray"
        )
        self.record_status_label.pack(anchor=tk.W, pady=2)
        ToolTip(
            recording,
            "Записывает видимое окно приложения в MP4. Рядом автоматически "
            "создаются CSV-журнал пиков и JSON-снимок настроек. Во время "
            "записи не сворачивайте и не перекрывайте окно другими окнами.",
        )

        hikvision = ttk.LabelFrame(controls, text="Hikvision: поиск по ИК-вспышке")
        hikvision.pack(fill=tk.X, pady=4)
        try:
            initial_rtsp = urllib_parse.urlsplit(self.rtsp_url_var.get())
            initial_api_url = (
                f"http://{initial_rtsp.hostname}" if initial_rtsp.hostname else ""
            )
        except ValueError:
            initial_api_url = ""
        api_row = ttk.Frame(hikvision)
        api_row.pack(fill=tk.X, pady=2)
        ttk.Label(api_row, text="ISAPI:").pack(side=tk.LEFT)
        self.hik_api_url_var = tk.StringVar(value=initial_api_url)
        ttk.Entry(api_row, textvariable=self.hik_api_url_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 2)
        )
        ttk.Button(
            api_row,
            text="↻",
            width=3,
            command=self._fill_hikvision_api_from_rtsp,
        ).pack(side=tk.RIGHT)
        auth_row = ttk.Frame(hikvision)
        auth_row.pack(fill=tk.X, pady=2)
        self.hik_user_var = tk.StringVar(value="")
        self.hik_password_var = tk.StringVar(value="")
        ttk.Label(auth_row, text="Логин:").pack(side=tk.LEFT)
        ttk.Entry(auth_row, textvariable=self.hik_user_var, width=9).pack(
            side=tk.LEFT, padx=(3, 6)
        )
        ttk.Label(auth_row, text="Пароль:").pack(side=tk.LEFT)
        ttk.Entry(
            auth_row,
            textvariable=self.hik_password_var,
            show="•",
            width=11,
        ).pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        channel_row = ttk.Frame(hikvision)
        channel_row.pack(fill=tk.X, pady=2)
        self.hik_channel_var = tk.IntVar(value=self.calibration.hikvision_channel)
        self.ir_settle_var = tk.DoubleVar(value=self.calibration.ir_settle_seconds)
        ttk.Label(channel_row, text="Канал:").pack(side=tk.LEFT)
        ttk.Spinbox(
            channel_row,
            from_=1,
            to=64,
            textvariable=self.hik_channel_var,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(channel_row, text="Стабилизация, с:").pack(side=tk.LEFT)
        ttk.Spinbox(
            channel_row,
            from_=0.5,
            to=10.0,
            increment=0.5,
            textvariable=self.ir_settle_var,
            width=6,
        ).pack(side=tk.RIGHT)
        flash_row = ttk.Frame(hikvision)
        flash_row.pack(fill=tk.X, pady=2)
        self.ir_flash_delta_var = tk.IntVar(value=self.calibration.ir_flash_delta)
        self.ir_search_scale_var = tk.DoubleVar(value=self.calibration.ir_search_scale)
        ttk.Label(flash_row, text="Мин. вспышка:").pack(side=tk.LEFT)
        ttk.Spinbox(
            flash_row,
            from_=5,
            to=200,
            textvariable=self.ir_flash_delta_var,
            width=6,
        ).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(flash_row, text="Масштаб поиска:").pack(side=tk.LEFT)
        ttk.Spinbox(
            flash_row,
            from_=1.0,
            to=12.0,
            increment=0.5,
            textvariable=self.ir_search_scale_var,
            width=6,
        ).pack(side=tk.RIGHT)
        lock_row = ttk.Frame(hikvision)
        lock_row.pack(fill=tk.X, pady=2)
        self.ir_lock_enabled_var = tk.BooleanVar(
            value=self.calibration.ir_lock_enabled
        )
        self.ir_lock_radius_var = tk.DoubleVar(
            value=self.calibration.ir_lock_radius
        )
        ttk.Checkbutton(
            lock_row,
            text="Удерживать ИК-метку",
            variable=self.ir_lock_enabled_var,
            command=self._on_ir_lock_change,
        ).pack(side=tk.LEFT)
        ir_lock_radius_spin = ttk.Spinbox(
            lock_row,
            from_=5,
            to=300,
            increment=5,
            textvariable=self.ir_lock_radius_var,
            width=6,
        )
        ir_lock_radius_spin.pack(side=tk.RIGHT)
        ir_lock_radius_spin.bind("<Return>", self._on_ir_lock_change)
        ir_lock_radius_spin.bind("<FocusOut>", self._on_ir_lock_change)
        ttk.Label(lock_row, text="Радиус, px:").pack(side=tk.RIGHT, padx=(3, 2))
        hik_buttons = ttk.Frame(hikvision)
        hik_buttons.pack(fill=tk.X, pady=2)
        ttk.Button(
            hik_buttons,
            text="Проверить",
            command=self.test_hikvision_isapi,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(
            hik_buttons,
            text="День",
            command=lambda: self.set_hikvision_mode("day"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(
            hik_buttons,
            text="Ночь",
            command=lambda: self.set_hikvision_mode("night"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(
            hik_buttons,
            text="Авто",
            command=lambda: self.set_hikvision_mode("auto"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))
        self.ir_scan_btn = ttk.Button(
            hikvision,
            text="🔦 Найти отражатели: день → ночь",
            command=self.start_ir_flash_scan,
        )
        self.ir_scan_btn.pack(fill=tk.X, pady=2)
        self.hik_status_label = ttk.Label(
            hikvision,
            text="Пустые логин/пароль берутся из RTSP URL",
            foreground="gray",
            wraplength=330,
        )
        self.hik_status_label.pack(fill=tk.X, pady=2)
        ToolTip(
            hikvision,
            "Процедура один раз получает кадры в режиме Day и Night, выделяет "
            "только появившийся ИК-отклик и переносит исходные области к "
            "найденным отражателям. После поиска камера остается в Night.",
        )

        detection = ttk.LabelFrame(controls, text="Обнаружение интегрального блика")
        detection.pack(fill=tk.X, pady=4)
        self.expected_var = tk.IntVar(value=self.calibration.expected_reflectors)
        expected_row = ttk.Frame(detection)
        expected_row.pack(fill=tk.X, pady=2)
        ttk.Label(expected_row, text="Количество пиков:").pack(side=tk.LEFT)
        expected_spin = ttk.Spinbox(
            expected_row,
            from_=1,
            to=20,
            textvariable=self.expected_var,
            width=6,
            command=self._on_expected_change,
        )
        expected_spin.pack(side=tk.RIGHT)
        expected_spin.bind("<Return>", self._on_expected_change)
        expected_spin.bind("<FocusOut>", self._on_expected_change)

        self.min_area_var = tk.IntVar(value=self.calibration.min_area)
        self.max_area_var = tk.IntVar(value=self.calibration.max_area)
        self.circularity_var = tk.DoubleVar(value=self.calibration.circularity_threshold)
        self.brightness_var = tk.IntVar(value=self.calibration.brightness_threshold)
        self.contrast_var = tk.IntVar(value=self.calibration.contrast_threshold)
        self.blur_var = tk.DoubleVar(value=self.calibration.blur_sigma)
        self.percentile_var = tk.DoubleVar(value=self.calibration.brightness_percentile)
        self.merge_radius_var = tk.IntVar(value=self.calibration.merge_radius)
        self.center_power_var = tk.DoubleVar(value=self.calibration.center_power)
        self.adaptive_var = tk.BooleanVar(value=self.calibration.adaptive_threshold)

        self._add_scale(detection, "Мин. площадь, px", self.min_area_var, 1, 500)
        self._add_scale(detection, "Макс. площадь, px", self.max_area_var, 50, 10000)
        self._add_scale(
            detection,
            "Ожидаемая округлость (оценка)",
            self.circularity_var,
            0.05,
            1.0,
            2,
            "Неправильная форма больше не отбраковывается. Параметр влияет только на оценку уверенности.",
        )
        self._add_scale(detection, "Мин. яркость", self.brightness_var, 0, 255)
        self._add_scale(detection, "Мин. локальный контраст", self.contrast_var, 1, 150)
        self._add_scale(detection, "Размытие sigma", self.blur_var, 0, 5, 2)
        ttk.Checkbutton(
            detection,
            text="Адаптивный порог при изменении освещения",
            variable=self.adaptive_var,
            command=self._on_param_change,
        ).pack(anchor=tk.W, pady=2)
        self._add_scale(
            detection,
            "Процентиль яркости",
            self.percentile_var,
            90.0,
            99.9,
            1,
        )
        self._add_scale(
            detection,
            "Радиус объединения, px",
            self.merge_radius_var,
            0,
            40,
            0,
            "Объединяет разорванные части одного блика от призмы.",
        )
        self._add_scale(
            detection,
            "Вес яркого ядра центра",
            self.center_power_var,
            1.0,
            4.0,
            2,
        )
        ttk.Button(
            detection,
            text="⚙ Индивидуальные настройки областей…",
            command=self.open_region_settings_dialog,
        ).pack(fill=tk.X, padx=2, pady=(5, 3))

        tracking = ttk.LabelFrame(controls, text="Устойчивость сопровождения")
        tracking.pack(fill=tk.X, pady=4)
        self.smoothing_var = tk.DoubleVar(value=self.calibration.smoothing_alpha)
        self.expand_step_var = tk.IntVar(value=self.calibration.roi_expand_step)
        self.max_scale_var = tk.DoubleVar(value=self.calibration.roi_max_scale)
        self.hold_frames_var = tk.IntVar(value=self.calibration.lost_hold_frames)
        self.max_jump_var = tk.DoubleVar(value=self.calibration.max_jump)
        self._add_scale(
            tracking,
            "Реакция фильтра центра",
            self.smoothing_var,
            0.02,
            1.0,
            2,
            "Меньше — центр спокойнее; больше — быстрее следует за перемещением.",
        )
        self._add_scale(
            tracking, "Расширение области, px/кадр", self.expand_step_var, 1, 30
        )
        self._add_scale(
            tracking, "Макс. увеличение области", self.max_scale_var, 1.0, 8.0, 1
        )
        self._add_scale(
            tracking, "Удержание при потере, кадров", self.hold_frames_var, 1, 300
        )
        self._add_scale(
            tracking, "Макс. скачок за кадр, px", self.max_jump_var, 5, 500
        )

        layers = ttk.LabelFrame(controls, text="Слои отображения")
        layers.pack(fill=tk.X, pady=4)
        self.show_points_var = tk.BooleanVar(value=self.calibration.show_points)
        self.show_circles_var = tk.BooleanVar(value=self.calibration.show_circles)
        self.show_frames_var = tk.BooleanVar(value=self.calibration.show_frames)
        self.show_lines_var = tk.BooleanVar(value=self.calibration.show_lines)
        self.show_distances_var = tk.BooleanVar(
            value=self.calibration.show_distances
        )
        self.show_distance_changes_var = tk.BooleanVar(
            value=self.calibration.show_distance_changes
        )
        self.close_shape_var = tk.BooleanVar(value=self.calibration.close_shape)
        layer_options = (
            ("Точки и подписи", self.show_points_var),
            ("Круги отражателей", self.show_circles_var),
            ("Рамки областей", self.show_frames_var),
            ("Линии между точками", self.show_lines_var),
            ("Расстояния по линиям", self.show_distances_var),
            ("Изменение расстояний", self.show_distance_changes_var),
            ("Замкнуть контур", self.close_shape_var),
        )
        for label, variable in layer_options:
            ttk.Checkbutton(
                layers,
                text=label,
                variable=variable,
                command=self._on_display_layer_change,
            ).pack(anchor=tk.W, padx=3, pady=1)
        ToolTip(
            layers,
            "Слои включаются независимо. Контур соединяет точки по номерам; "
            "для замыкания требуется не менее трех активных точек. "
            "dL показывает изменение длины относительно предыдущего кадра.",
        )

        preset_frame = ttk.LabelFrame(controls, text="Пресеты")
        preset_frame.pack(fill=tk.X, pady=4)
        self.preset_var = tk.StringVar(value="Базовый")
        self.preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=self._all_preset_names(),
            state="readonly",
        )
        self.preset_combo.pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(
            preset_frame,
            text="Применить выбранный пресет",
            command=self.apply_selected_preset,
        ).pack(fill=tk.X, padx=2, pady=2)
        preset_manage = ttk.Frame(preset_frame)
        preset_manage.pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(
            preset_manage,
            text="Создать из текущих…",
            command=self.create_user_preset,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(
            preset_manage,
            text="Удалить",
            command=self.delete_user_preset,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))
        preset_buttons = ttk.Frame(preset_frame)
        preset_buttons.pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(
            preset_buttons,
            text="Экспорт JSON…",
            command=self.save_preset_file,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(
            preset_buttons,
            text="Импорт JSON…",
            command=self.load_preset_file,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        io_frame = ttk.LabelFrame(controls, text="Настройки")
        io_frame.pack(fill=tk.X, pady=4)
        ttk.Button(
            io_frame, text="💾 Сохранить", command=self.save_calibration
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        ttk.Button(
            io_frame, text="📂 Загрузить", command=self.load_calibration
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)

        display_frame = ttk.Frame(main)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(display_frame, bg="black", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<MouseWheel>", self._zoom_wheel)
        self.main_canvas.bind("<Button-4>", self._zoom_wheel)
        self.main_canvas.bind("<Button-5>", self._zoom_wheel)
        self.main_canvas.bind("<ButtonPress-1>", self._start_pan)
        self.main_canvas.bind("<B1-Motion>", self._pan_view)
        self.main_canvas.bind("<ButtonRelease-1>", self._end_pan)
        self.main_canvas.bind("<Double-Button-1>", lambda event: self.reset_view_zoom())

        status = ttk.Frame(self.root)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(status, text="Готов к работе", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fps_label = ttk.Label(status, text="FPS: 0", relief=tk.SUNKEN, width=12)
        self.fps_label.pack(side=tk.RIGHT)
        self._on_source_change()
        self._update_region_status()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить настройки", command=self.save_calibration)
        file_menu.add_command(label="Загрузить настройки", command=self.load_calibration)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить отдельный пресет", command=self.save_preset_file)
        file_menu.add_command(label="Загрузить отдельный пресет", command=self.load_preset_file)
        file_menu.add_separator()
        file_menu.add_command(label="Начать/остановить запись окна", command=self.toggle_window_recording)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._on_closing)
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Вид", menu=view_menu)
        self.show_preprocess_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(
            label="Показать яркостную предобработку",
            variable=self.show_preprocess_var,
            command=self._toggle_preprocess,
        )
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="Инструкция", command=self._show_help)
        help_menu.add_command(label="О программе", command=self._show_about)

    def _show_help(self):
        text = """ПОРЯДОК РАБОТЫ

1. Выберите RTSP или локальную камеру.
2. Включите «Режим калибровки» — появится живой кадр.
3. Укажите требуемое количество пиков.
4. Нажмите «Задать области всех пиков» и поочередно обведите каждый отражатель.
   Внутри одной области должен находиться один ожидаемый отражатель.
5. Настройте яркость, контраст и радиус объединения частей блика.
6. Запустите отслеживание.

ПРЕСЕТЫ
«Базовый» содержит начальные параметры программы. Встроенный пресет выбирается
из списка и применяется кнопкой. «Создать из текущих…» добавляет именованный
пользовательский пресет, который остается в списке после перезапуска программы.
Пользовательские пресеты можно удалять, экспортировать в отдельный JSON и
импортировать обратно. Области пиков и индивидуальные профили в пресет не
входят; они сохраняются кнопкой «Сохранить» как полные настройки.

ЗАПИСЬ НАБЛЮДЕНИЯ
1. Запустите видеопоток и отслеживание.
2. Выберите частоту записи и нажмите «Начать запись окна».
3. Укажите MP4-файл. Рядом автоматически появятся файлы *_states.csv и
   *_settings.json.
4. Не сворачивайте окно, не блокируйте экран и не перекрывайте приложение.
5. Для завершения нажмите «Остановить запись».

ПРИБЛИЖЕНИЕ ПРОСМОТРА
• колесо мыши над изображением — приблизить или отдалить;
• левая кнопка мыши с перетаскиванием — переместить увеличенный кадр;
• двойной щелчок или кнопка «Сброс» — вернуть масштаб 1:1.
Масштаб меняет только просмотр: координаты вычисляются по исходному кадру.

СЛОИ ОТОБРАЖЕНИЯ
Точки, круги, рамки, линии, расстояния и изменение расстояний можно включать
независимо. Точки соединяются по порядку P1—P2—P3... Флажок «Замкнуть контур»
добавляет линию от последней активной точки к P1, если точек не меньше трех.
dL — изменение длины данного ребра относительно предыдущего обработанного кадра.

ИНДИВИДУАЛЬНЫЕ НАСТРОЙКИ ОБЛАСТЕЙ
Кнопка в разделе обнаружения открывает параметры выбранной области Pn.
Включите индивидуальный режим, задайте значения и нажмите «Применить к
выбранной». Можно скопировать глобальные значения, применить один профиль ко
всем областям или вернуть выбранную/все области к глобальным параметрам.
Надпись IND возле рамки означает, что область использует отдельный профиль.

ПОИСК HIKVISION ПО ИК-ВСПЫШКЕ
1. Укажите примерные области P1…Pn и запустите живой RTSP-поток.
2. Проверьте адрес ISAPI, канал и авторизацию кнопкой «Проверить».
   Пустые логин и пароль автоматически берутся из RTSP URL.
3. Нажмите «Найти отражатели: день → ночь». Камера последовательно включит
   Day и Night. Программа ищет объекты, которые были темными в Day и стали
   яркими в Night; белые объекты, светлые в обоих режимах, исключаются.
4. Зеленые метки — найденные ИК-отражатели; исходные области автоматически
   переносятся к ним. Оранжевые рамки — остальные кандидаты.
5. Если отражатель не найден, увеличьте «Масштаб поиска» (например, с 4 до 6)
   либо уменьшите «Мин. вспышка» (например, с 25 до 15).
После процедуры камера остается в Night для обычного сопровождения. Постоянное
переключение механического IR-cut фильтра во время каждого кадра не выполняется.
Флажок «Удерживать ИК-метку» не позволяет треку перескочить на более яркий
объект за пределами заданного радиуса. Рекомендуемый радиус — 20–40 px.

LOCK — пик найден на текущем кадре.
HOLD — блик временно не распознан, но последняя точка удерживается.
LOST — превышено заданное число кадров удержания.

При HOLD/LOST поиск остается привязанным к исходной области и автоматически
расширяется максимум примерно до двойного размера. Если отражатель найден за
исходной границей, область сохраняет достаточный размер для его сопровождения.
Области и новые параметры сохраняются в JSON вместе с остальными настройками.

После изменения любого параметра обнаружения или сопровождения детектор,
траектории, скорости, история и сглаживание полностью сбрасываются. Следующий
кадр обрабатывается с нулевого состояния, при этом заданные области и фон
сохраняются.
"""
        window = tk.Toplevel(self.root)
        window.title("Инструкция")
        window.geometry("720x680")
        widget = tk.Text(window, wrap=tk.WORD, padx=12, pady=12)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)
        widget.pack(fill=tk.BOTH, expand=True)

    def _show_about(self):
        messagebox.showinfo(
            "О программе",
            "Система обнаружения отражателей v5.6\n\n"
            "• один постоянный трек на одну заданную область\n"
            "• объединение сложного блика в единую точку\n"
            "• яркостно-взвешенный субпиксельный центр\n"
            "• адаптивное расширение области при потере\n"
            "• масштабирование и перемещение изображения\n"
            "• запись окна в MP4 и журнал состояний CSV\n"
            "• встроенные и постоянные пользовательские пресеты\n"
            "• закреплённый поиск с расширением до двойной области\n"
            "• независимые слои геометрии и покадровые изменения расстояний\n"
            "• индивидуальные настройки обнаружения для P1…Pn\n"
            "• ИК-поиск Hikvision по признаку «темный Day → яркий Night»\n"
            "• ИК-якорь против перехода на посторонние белые объекты",
        )

    def _on_source_change(self):
        self.use_rtsp = self.source_type_var.get() == "rtsp"
        if self.use_rtsp:
            self.local_frame.pack_forget()
            self.rtsp_frame.pack(fill=tk.X, pady=3)
        else:
            self.rtsp_frame.pack_forget()
            self.local_frame.pack(fill=tk.X, pady=3)

    def _toggle_preprocess(self):
        self.show_preprocessing = self.show_preprocess_var.get()

    def _fill_hikvision_api_from_rtsp(self):
        try:
            parsed = urllib_parse.urlsplit(self.rtsp_url_var.get().strip())
            if not parsed.hostname:
                raise ValueError("в RTSP URL отсутствует адрес камеры")
            self.hik_api_url_var.set(f"http://{parsed.hostname}")
            self.hik_status_label.config(
                text="Адрес ISAPI взят из RTSP URL", foreground="gray"
            )
        except Exception as exc:
            messagebox.showerror("Hikvision ISAPI", str(exc))

    def _hikvision_connection_parameters(self):
        try:
            parsed = urllib_parse.urlsplit(self.rtsp_url_var.get().strip())
        except ValueError as exc:
            raise ValueError(f"Неверный RTSP URL: {exc}") from exc
        api_url = self.hik_api_url_var.get().strip()
        if not api_url and parsed.hostname:
            api_url = f"http://{parsed.hostname}"
            self.hik_api_url_var.set(api_url)
        username = self.hik_user_var.get().strip()
        password = self.hik_password_var.get()
        if not username and parsed.username:
            username = urllib_parse.unquote(parsed.username)
        if not password and parsed.password:
            password = urllib_parse.unquote(parsed.password)
        if not username:
            username = "admin"
        return api_url, username, password

    def _make_hikvision_control(self) -> HikvisionISAPI:
        api_url, username, password = self._hikvision_connection_parameters()
        try:
            channel = max(1, int(self.hik_channel_var.get()))
        except (tk.TclError, ValueError):
            channel = 1
            self.hik_channel_var.set(channel)
        self.calibration.hikvision_channel = channel
        return HikvisionISAPI(api_url, username, password, channel=channel)

    def _background_call(self, worker, on_success, on_error=None):
        """Выполняет HTTP/тяжелую операцию без блокировки окна Tk."""
        result_queue = queue.Queue(maxsize=1)

        def run_worker():
            try:
                result_queue.put((True, worker()))
            except Exception as exc:
                result_queue.put((False, exc))

        threading.Thread(target=run_worker, daemon=True).start()

        def poll_result():
            try:
                succeeded, payload = result_queue.get_nowait()
            except queue.Empty:
                self.root.after(50, poll_result)
                return
            if succeeded:
                on_success(payload)
            elif on_error is not None:
                on_error(payload)
            else:
                messagebox.showerror("Hikvision ISAPI", str(payload))

        self.root.after(50, poll_result)

    def test_hikvision_isapi(self):
        if self._ir_scan_running:
            self.status_label.config(text="Дождитесь завершения ИК-поиска")
            return
        try:
            control = self._make_hikvision_control()
            self.hik_status_label.config(
                text="Проверка ISAPI…", foreground="#b05a00"
            )
        except Exception as exc:
            messagebox.showerror("Hikvision ISAPI", str(exc))
            return

        def worker():
            info = control.get_device_info()
            mode, _, _ = control.get_ircut()
            return info, mode

        def success(result):
            info, mode = result
            self.hikvision_control = control
            model = info.get("model") or info.get("deviceName") or "Hikvision"
            self.hik_status_label.config(
                text=f"Связь установлена: {model}; Day/Night = {mode}",
                foreground="green",
            )

        def failure(exc):
            self.hik_status_label.config(text="Ошибка ISAPI", foreground="red")
            messagebox.showerror("Hikvision ISAPI", str(exc))

        self._background_call(worker, success, failure)

    def set_hikvision_mode(self, mode: str):
        if self._ir_scan_running:
            self.status_label.config(text="Дождитесь завершения ИК-поиска")
            return
        try:
            control = self._make_hikvision_control()
            self.hik_status_label.config(
                text=f"Переключение камеры: {mode}…", foreground="#b05a00"
            )
        except Exception as exc:
            messagebox.showerror("Hikvision ISAPI", str(exc))
            return

        def success(old_mode):
            self.hikvision_control = control
            self.hik_status_label.config(
                text=f"Установлен режим {mode}; ранее: {old_mode}",
                foreground="green",
            )

        def failure(exc):
            self.hik_status_label.config(text="Ошибка ISAPI", foreground="red")
            messagebox.showerror("Hikvision ISAPI", str(exc))

        self._background_call(lambda: control.set_ircut(mode), success, failure)

    def start_ir_flash_scan(self):
        if self._ir_scan_running:
            return
        if not self.use_rtsp:
            messagebox.showwarning(
                "ИК-поиск", "Управление Day/Night доступно только для RTSP-камеры."
            )
            return
        if self.current_frame is None or not (self.is_running or self.is_calibrating):
            messagebox.showwarning(
                "ИК-поиск",
                "Сначала включите калибровку или отслеживание и дождитесь кадра.",
            )
            return
        if not self.calibration.peak_regions:
            messagebox.showwarning(
                "ИК-поиск",
                "Сначала задайте примерные области P1…Pn. ИК-поиск может "
                "перенести их к отражателям, но должен знать ожидаемое положение.",
            )
            return
        try:
            self._update_ir_settings_from_ui()
            control = self._make_hikvision_control()
        except Exception as exc:
            messagebox.showerror("ИК-поиск", str(exc))
            return

        self.hikvision_control = control
        self._ir_scan_running = True
        self._ir_scan_token += 1
        token = self._ir_scan_token
        self._ir_day_samples = []
        self._ir_night_samples = []
        self.ir_scan_btn.config(state=tk.DISABLED, text="Переключение в Day…")
        self.hik_status_label.config(
            text="Этап 1/4: включение дневного режима", foreground="#b05a00"
        )

        def prepare_day():
            original, _, _ = control.get_ircut()
            control.set_ircut("day")
            return original

        def day_ready(original_mode):
            if token != self._ir_scan_token:
                return
            self._ir_original_mode = original_mode or "auto"
            delay = int(round(self.calibration.ir_settle_seconds * 1000))
            self.ir_scan_btn.config(text="Ожидание стабилизации Day…")
            self.hik_status_label.config(
                text=f"Этап 2/4: стабилизация Day ({delay / 1000:.1f} с)",
                foreground="#b05a00",
            )
            self.root.after(
                delay,
                lambda: self._collect_ir_samples(
                    self._ir_day_samples,
                    token,
                    self._switch_ir_scan_to_night,
                ),
            )

        self._background_call(
            prepare_day,
            day_ready,
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=False),
        )

    def _collect_ir_samples(self, target: List[np.ndarray], token: int, done):
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        if self.current_frame is not None:
            target.append(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY))
        required = max(3, int(self.calibration.ir_sample_count))
        if len(target) >= required:
            done(token)
            return
        self.root.after(140, lambda: self._collect_ir_samples(target, token, done))

    def _switch_ir_scan_to_night(self, token: int):
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        self.ir_scan_btn.config(text="Переключение в Night…")
        self.hik_status_label.config(
            text="Этап 3/4: включение ИК-подсветки и Night", foreground="#b05a00"
        )

        def night_ready(_):
            if token != self._ir_scan_token:
                return
            delay = int(round(self.calibration.ir_settle_seconds * 1000))
            self.ir_scan_btn.config(text="Ожидание стабилизации Night…")
            self.root.after(
                delay,
                lambda: self._collect_ir_samples(
                    self._ir_night_samples,
                    token,
                    self._analyze_ir_scan_async,
                ),
            )

        self._background_call(
            lambda: self.hikvision_control.set_ircut("night"),
            night_ready,
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=True),
        )

    def _analyze_ir_scan_async(self, token: int):
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        self.ir_scan_btn.config(text="Анализ ИК-отклика…")
        self.hik_status_label.config(
            text="Этап 4/4: выделение вспышек и привязка P1…Pn",
            foreground="#b05a00",
        )
        day_samples = list(self._ir_day_samples)
        night_samples = list(self._ir_night_samples)
        self._background_call(
            lambda: self._analyze_ir_flash_frames(day_samples, night_samples),
            lambda result: self._finish_ir_flash_scan(token, result),
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=False),
        )

    @staticmethod
    def _align_ir_day_frame(day: np.ndarray, night: np.ndarray):
        """Совмещает Day с Night по фону; при неудаче оставляет кадр как есть."""
        try:
            height, width = day.shape
            registration_scale = min(1.0, 1000.0 / max(height, width))
            if registration_scale < 1.0:
                registration_size = (
                    max(64, int(round(width * registration_scale))),
                    max(64, int(round(height * registration_scale))),
                )
                day_registration = cv2.resize(
                    day, registration_size, interpolation=cv2.INTER_AREA
                )
                night_registration = cv2.resize(
                    night, registration_size, interpolation=cv2.INTER_AREA
                )
            else:
                day_registration = day
                night_registration = night

            def normalized_for_registration(image):
                low, high = np.percentile(image, (5.0, 95.0))
                span = max(1.0, float(high - low))
                normalized = np.clip((image - low) / span, 0.0, 1.0)
                return cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), 1.2)

            day_registration = normalized_for_registration(day_registration)
            night_registration = normalized_for_registration(night_registration)
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                80,
                1e-5,
            )
            correlation, warp = cv2.findTransformECC(
                night_registration,
                day_registration,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
            )
            shift_x = float(warp[0, 2] / registration_scale)
            shift_y = float(warp[1, 2] / registration_scale)
            if (
                not np.isfinite(correlation)
                or correlation < 0.20
                or abs(shift_x) > 30.0
                or abs(shift_y) > 30.0
            ):
                return day, (0.0, 0.0, float(correlation))
            full_warp = np.array(
                [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
                dtype=np.float32,
            )
            aligned = cv2.warpAffine(
                day,
                full_warp,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, (shift_x, shift_y, float(correlation))
        except Exception:
            return day, (0.0, 0.0, 0.0)

    def _analyze_ir_flash_frames(
        self, day_samples: List[np.ndarray], night_samples: List[np.ndarray]
    ):
        if not day_samples or not night_samples:
            raise RuntimeError("Не удалось получить стабильные кадры Day/Night")
        day = np.median(np.stack(day_samples), axis=0).astype(np.float32)
        night = np.median(np.stack(night_samples), axis=0).astype(np.float32)
        if day.shape != night.shape:
            raise RuntimeError("Размер кадра изменился при переключении Day/Night")

        day, alignment = self._align_ir_day_frame(day, night)

        # Сравниваем не абсолютные уровни, а фотометрически нормированный
        # ИК-прирост. Белый объект, светлый в обоих режимах, получает почти
        # нулевой вес. Темная в Day и яркая в Night ретрометка — высокий вес.
        day_scale = max(20.0, float(np.percentile(day, 95.0)))
        night_scale = max(20.0, float(np.percentile(night, 95.0)))
        day_normalized = np.clip(day / day_scale, 0.0, 1.5)
        night_normalized = np.clip(night / night_scale, 0.0, 1.5)

        day_local = day_normalized - cv2.GaussianBlur(
            day_normalized, (0, 0), 15.0
        )
        night_local = night_normalized - cv2.GaussianBlur(
            night_normalized, (0, 0), 15.0
        )
        absolute_gain = np.maximum(night_normalized - day_normalized, 0.0)
        relative_gain = np.maximum(
            np.log((night_normalized + 0.06) / (day_normalized + 0.06)),
            0.0,
        )
        local_gain = np.maximum(night_local - day_local, 0.0)
        day_dark_weight = np.clip((0.82 - day_normalized) / 0.52, 0.0, 1.0)
        night_bright_weight = np.clip(
            (night_normalized - 0.35) / 0.45, 0.0, 1.0
        )
        response = (
            70.0 * absolute_gain
            + 48.0 * relative_gain
            + 85.0 * local_gain
        ) * day_dark_weight * night_bright_weight
        response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), 0.8)
        positive_values = response[response > 0]
        percentile_level = (
            float(np.percentile(positive_values, 99.2))
            if positive_values.size
            else 255.0
        )
        # Процентиль не должен оставлять только самый сильный отражатель:
        # несколько призм могут отличаться по яркости в разы. Ограничиваем
        # автоматический уровень долей глобального максимума, а ложные
        # кандидаты затем отсекаются ожидаемыми областями P1...Pn.
        maximum_response = float(np.max(response)) if response.size else 0.0
        adaptive_level = min(percentile_level, maximum_response * 0.35)
        threshold = max(float(self.calibration.ir_flash_delta), adaptive_level)
        binary = np.where(response >= threshold, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        minimum_area = max(2, int(self.calibration.min_area // 2))
        maximum_area = max(300, int(self.calibration.max_area * 6))
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area or area > maximum_area:
                continue
            component = labels == label
            ys, xs = np.where(component)
            response_values = response[ys, xs]
            peak_response = float(np.max(response_values))
            night_peak = float(np.max(night[component]))
            day_level = float(np.mean(day_normalized[component]))
            night_level = float(np.mean(night_normalized[component]))
            if night_peak < max(90.0, self.calibration.brightness_threshold * 0.55):
                continue
            if day_level >= 0.82 or night_level <= day_level + 0.12:
                continue
            weights = np.power(response_values, 2.0)
            total = float(np.sum(weights))
            if total <= 0:
                continue
            center = (
                float(np.sum(xs * weights) / total),
                float(np.sum(ys * weights) / total),
            )
            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            strength = peak_response * np.sqrt(area)
            candidates.append(
                {
                    "position": center,
                    "bbox": bbox,
                    "area": area,
                    "response": peak_response,
                    "strength": float(strength),
                    "day_level": day_level,
                    "night_level": night_level,
                }
            )
        candidates.sort(key=lambda item: item["strength"], reverse=True)
        candidates = candidates[: max(20, len(self.calibration.peak_regions) * 8)]

        pairs = []
        search_scale = max(1.0, float(self.calibration.ir_search_scale))
        for region_index, region in enumerate(self.calibration.peak_regions):
            x, y, width, height = region
            center = np.array([x + width / 2.0, y + height / 2.0])
            half_width = width * search_scale / 2.0
            half_height = height * search_scale / 2.0
            normalizer = max(10.0, float(np.hypot(width, height)))
            for candidate_index, candidate in enumerate(candidates):
                px, py = candidate["position"]
                if not (
                    center[0] - half_width <= px <= center[0] + half_width
                    and center[1] - half_height <= py <= center[1] + half_height
                ):
                    continue
                distance = float(np.linalg.norm(np.array([px, py]) - center))
                response_bonus = min(0.30, candidate["response"] / 255.0 * 0.30)
                cost = distance / normalizer - response_bonus
                pairs.append((cost, region_index, candidate_index))
        pairs.sort(key=lambda item: item[0])

        used_regions = set()
        used_candidates = set()
        matches = []
        for _, region_index, candidate_index in pairs:
            if region_index in used_regions or candidate_index in used_candidates:
                continue
            used_regions.add(region_index)
            used_candidates.add(candidate_index)
            matches.append((region_index, candidates[candidate_index]))

        night_u8 = np.clip(night, 0, 255).astype(np.uint8)
        preview = cv2.cvtColor(night_u8, cv2.COLOR_GRAY2BGR)
        try:
            response_max = max(1.0, float(np.max(response)))
            response_u8 = np.clip(response / response_max * 255.0, 0, 255).astype(
                np.uint8
            )
            heatmap = cv2.applyColorMap(response_u8, cv2.COLORMAP_TURBO)
            preview = cv2.addWeighted(preview, 0.78, heatmap, 0.22, 0.0)
        except Exception:
            pass
        for candidate in candidates:
            x, y, width, height = candidate["bbox"]
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                (0, 165, 255),
                1,
            )
        for region_index, candidate in matches:
            px, py = [int(round(value)) for value in candidate["position"]]
            cv2.circle(preview, (px, py), 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                preview,
                f"P{region_index + 1} IR",
                (px + 12, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            preview,
            (
                f"IR dark-to-bright: {len(matches)}/{len(self.calibration.peak_regions)}; "
                f"T={threshold:.1f}; shift=({alignment[0]:+.1f},{alignment[1]:+.1f})"
            ),
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return matches, candidates, preview, threshold

    def _finish_ir_flash_scan(self, token: int, result):
        if token != self._ir_scan_token:
            return
        matches, candidates, preview, threshold = result
        confirmed_centers = {
            str(region_index + 1): [
                float(candidate["position"][0]),
                float(candidate["position"][1]),
            ]
            for region_index, candidate in matches
        }
        self.calibration.ir_confirmed_centers = confirmed_centers
        if matches:
            regions = list(self.calibration.peak_regions)
            frame_shape = preview.shape
            for region_index, candidate in matches:
                _, _, old_width, old_height = regions[region_index]
                bbox_width = candidate["bbox"][2]
                bbox_height = candidate["bbox"][3]
                margin = 12
                width = max(old_width, bbox_width + 2 * margin)
                height = max(old_height, bbox_height + 2 * margin)
                px, py = candidate["position"]
                moved = clip_rect(
                    (
                        int(round(px - width / 2.0)),
                        int(round(py - height / 2.0)),
                        width,
                        height,
                    ),
                    frame_shape,
                )
                if moved is not None:
                    regions[region_index] = moved
            self.calibration.peak_regions = regions
            xs = [region[0] for region in regions]
            ys = [region[1] for region in regions]
            x2s = [region[0] + region[2] for region in regions]
            y2s = [region[1] + region[3] for region in regions]
            self.calibration.roi_region = (
                min(xs),
                min(ys),
                max(x2s) - min(xs),
                max(y2s) - min(ys),
            )
            if self.tracker:
                self._reset_processing_modules(announce=False)
            self._update_region_status()

        self._ir_preview_overlay = preview
        self._ir_preview_until = time.time() + 6.0
        self._ir_scan_running = False
        self._ir_day_samples = []
        self._ir_night_samples = []
        self.ir_scan_btn.config(
            state=tk.NORMAL, text="🔦 Найти отражатели: день → ночь"
        )
        found = len(matches)
        expected = len(self.calibration.peak_regions)
        color = "green" if found == expected else "#b05a00"
        self.hik_status_label.config(
            text=(
                f"ИК-поиск: найдено {found}/{expected}; порог {threshold:.1f}. "
                "Камера оставлена в Night. Зеленые метки видны 6 с."
            ),
            foreground=color,
        )
        if found < expected:
            self.status_label.config(
                text=(
                    f"ИК-поиск нашел {found}/{expected}. Для пропущенной точки "
                    "увеличьте «Масштаб поиска» или уменьшите «Мин. вспышка»."
                )
            )
        else:
            self.status_label.config(
                text="Все области перенесены к ИК-отражателям; поиск начат заново"
            )

    def _abort_ir_flash_scan(self, error, restore_mode: bool):
        original_mode = self._ir_original_mode
        self._ir_scan_token += 1
        self._ir_scan_running = False
        self._ir_day_samples = []
        self._ir_night_samples = []
        self.ir_scan_btn.config(
            state=tk.NORMAL, text="🔦 Найти отражатели: день → ночь"
        )
        self.hik_status_label.config(text="ИК-поиск прерван", foreground="red")
        if restore_mode and self.hikvision_control is not None:
            self._background_call(
                lambda: self.hikvision_control.set_ircut(original_mode),
                lambda _: None,
                lambda _: None,
            )
        messagebox.showerror("ИК-поиск", str(error))

    def _update_ir_settings_from_ui(self):
        try:
            settle = float(str(self.ir_settle_var.get()).replace(",", "."))
            flash_delta = int(round(float(self.ir_flash_delta_var.get())))
            search_scale = float(
                str(self.ir_search_scale_var.get()).replace(",", ".")
            )
            lock_radius = float(
                str(self.ir_lock_radius_var.get()).replace(",", ".")
            )
            sample_count = int(self.calibration.ir_sample_count)
        except (tk.TclError, ValueError) as exc:
            raise ValueError("Проверьте числовые параметры ИК-поиска") from exc
        self.calibration.ir_settle_seconds = float(np.clip(settle, 0.5, 10.0))
        self.calibration.ir_flash_delta = int(np.clip(flash_delta, 5, 200))
        self.calibration.ir_search_scale = float(np.clip(search_scale, 1.0, 12.0))
        self.calibration.ir_sample_count = int(np.clip(sample_count, 3, 9))
        self.calibration.ir_lock_enabled = bool(self.ir_lock_enabled_var.get())
        self.calibration.ir_lock_radius = float(np.clip(lock_radius, 5.0, 300.0))
        self.ir_settle_var.set(self.calibration.ir_settle_seconds)
        self.ir_flash_delta_var.set(self.calibration.ir_flash_delta)
        self.ir_search_scale_var.set(self.calibration.ir_search_scale)
        self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)

    def _on_ir_lock_change(self, event=None):
        try:
            radius = float(str(self.ir_lock_radius_var.get()).replace(",", "."))
        except (tk.TclError, ValueError):
            radius = self.calibration.ir_lock_radius
        self.calibration.ir_lock_enabled = bool(self.ir_lock_enabled_var.get())
        self.calibration.ir_lock_radius = float(np.clip(radius, 5.0, 300.0))
        self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)
        if self.tracker:
            self._reset_processing_modules(announce=False)
        state = "включено" if self.calibration.ir_lock_enabled else "выключено"
        self.status_label.config(
            text=f"Удержание подтверждённой ИК-метки {state}; поиск начат заново"
        )

    def _on_display_layer_change(self):
        """Применяет слои без сброса детектора и накопленной геометрии."""
        self.calibration.show_points = bool(self.show_points_var.get())
        self.calibration.show_circles = bool(self.show_circles_var.get())
        self.calibration.show_frames = bool(self.show_frames_var.get())
        self.calibration.show_lines = bool(self.show_lines_var.get())
        self.calibration.show_distances = bool(self.show_distances_var.get())
        self.calibration.show_distance_changes = bool(
            self.show_distance_changes_var.get()
        )
        self.calibration.close_shape = bool(self.close_shape_var.get())
        if hasattr(self, "status_label"):
            self.status_label.config(text="Слои отображения обновлены")

    def _global_detection_settings(self) -> Dict:
        self._update_calibration_from_ui()
        return {
            name: getattr(self.calibration, name)
            for name in REGION_DETECTION_FIELDS
        }

    def _region_dialog_count(self) -> int:
        saved_ids = []
        for key in self.calibration.region_settings:
            try:
                saved_ids.append(int(key))
            except (TypeError, ValueError):
                pass
        return max(
            1,
            int(self.calibration.expected_reflectors),
            len(self.calibration.peak_regions),
            max(saved_ids, default=0),
        )

    def _region_dialog_selected_id(self) -> int:
        value = (
            self.region_dialog_selector_var.get()
            if self.region_dialog_selector_var is not None
            else "P1"
        )
        match = re.fullmatch(r"P(\d+)", str(value).strip())
        return int(match.group(1)) if match else 1

    def _close_region_settings_dialog(self):
        window = self.region_settings_window
        self.region_settings_window = None
        self.region_dialog_selector_var = None
        self.region_dialog_enabled_var = None
        self.region_dialog_vars = {}
        self.region_dialog_edit_widgets = []
        if window is not None and window.winfo_exists():
            window.destroy()

    def _refresh_region_dialog_selector(self):
        if (
            self.region_settings_window is None
            or not self.region_settings_window.winfo_exists()
            or self.region_dialog_selector_var is None
        ):
            return
        values = [f"P{index}" for index in range(1, self._region_dialog_count() + 1)]
        self.region_dialog_selector.config(values=values)
        if self.region_dialog_selector_var.get() not in values:
            self.region_dialog_selector_var.set(values[0])
        self._load_region_settings_to_dialog()

    def open_region_settings_dialog(self):
        if (
            self.region_settings_window is not None
            and self.region_settings_window.winfo_exists()
        ):
            self._refresh_region_dialog_selector()
            self.region_settings_window.lift()
            self.region_settings_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self.region_settings_window = window
        window.title("Индивидуальные настройки областей")
        window.geometry("570x590")
        window.minsize(520, 520)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_region_settings_dialog)

        body = ttk.Frame(window, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        selector_row = ttk.Frame(body)
        selector_row.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(selector_row, text="Область:").pack(side=tk.LEFT)
        self.region_dialog_selector_var = tk.StringVar(value="P1")
        self.region_dialog_selector = ttk.Combobox(
            selector_row,
            textvariable=self.region_dialog_selector_var,
            values=[
                f"P{index}" for index in range(1, self._region_dialog_count() + 1)
            ],
            width=10,
            state="readonly",
        )
        self.region_dialog_selector.pack(side=tk.LEFT, padx=7)
        self.region_dialog_selector.bind(
            "<<ComboboxSelected>>",
            lambda event: self._load_region_settings_to_dialog(),
        )

        self.region_dialog_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="Использовать отдельные параметры обнаружения для этой области",
            variable=self.region_dialog_enabled_var,
            command=self._toggle_region_dialog_fields,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            body,
            text=(
                "Сопровождение и расширение области остаются общими. "
                "Отдельно задаются параметры выделения самого блика."
            ),
            wraplength=525,
            foreground="#555555",
        ).pack(fill=tk.X, pady=(0, 8))

        fields = ttk.LabelFrame(body, text="Параметры обнаружения", padding=7)
        fields.pack(fill=tk.BOTH, expand=True)
        self.region_dialog_vars = {}
        self.region_dialog_edit_widgets = []
        row = 0
        for name in REGION_DETECTION_FIELDS:
            if name == "adaptive_threshold":
                variable = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(
                    fields,
                    text="Адаптивный порог при изменении освещения",
                    variable=variable,
                )
                widget.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=3)
            else:
                label = REGION_FIELD_META[name][0]
                ttk.Label(fields, text=label).grid(
                    row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3
                )
                variable = tk.StringVar()
                widget = ttk.Entry(fields, textvariable=variable, width=18)
                widget.grid(row=row, column=1, sticky=tk.EW, pady=3)
            self.region_dialog_vars[name] = variable
            self.region_dialog_edit_widgets.append(widget)
            row += 1
        fields.columnconfigure(1, weight=1)

        first_buttons = ttk.Frame(body)
        first_buttons.pack(fill=tk.X, pady=(9, 2))
        ttk.Button(
            first_buttons,
            text="Вставить глобальные",
            command=self._copy_global_to_region_dialog,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(
            first_buttons,
            text="Применить к выбранной",
            command=self._apply_region_settings_dialog,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        second_buttons = ttk.Frame(body)
        second_buttons.pack(fill=tk.X, pady=2)
        ttk.Button(
            second_buttons,
            text="Применить ко всем",
            command=lambda: self._apply_region_settings_dialog(apply_to_all=True),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(
            second_buttons,
            text="Глобальные для выбранной",
            command=self._disable_selected_region_settings,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        third_buttons = ttk.Frame(body)
        third_buttons.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(
            third_buttons,
            text="Сбросить индивидуальные у всех",
            command=self._clear_all_region_settings,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(
            third_buttons,
            text="Закрыть",
            command=self._close_region_settings_dialog,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self._load_region_settings_to_dialog()

    def _load_region_settings_to_dialog(self):
        if self.region_dialog_enabled_var is None or not self.region_dialog_vars:
            return
        region_id = self._region_dialog_selected_id()
        individual = self.calibration.region_settings.get(str(region_id))
        values = self._global_detection_settings()
        if individual:
            values.update(individual)
        self.region_dialog_enabled_var.set(bool(individual))
        for name, variable in self.region_dialog_vars.items():
            value = values[name]
            if name == "adaptive_threshold":
                variable.set(bool(value))
            elif REGION_FIELD_META[name][1] is int:
                variable.set(str(int(round(float(value)))))
            else:
                variable.set(f"{float(value):g}")
        self._toggle_region_dialog_fields()

    def _toggle_region_dialog_fields(self):
        enabled = bool(
            self.region_dialog_enabled_var
            and self.region_dialog_enabled_var.get()
        )
        state = "normal" if enabled else "disabled"
        for widget in self.region_dialog_edit_widgets:
            widget.configure(state=state)

    def _copy_global_to_region_dialog(self):
        if self.region_dialog_enabled_var is None:
            return
        self.region_dialog_enabled_var.set(True)
        values = self._global_detection_settings()
        for name, variable in self.region_dialog_vars.items():
            if name == "adaptive_threshold":
                variable.set(bool(values[name]))
            elif REGION_FIELD_META[name][1] is int:
                variable.set(str(int(round(float(values[name])))))
            else:
                variable.set(f"{float(values[name]):g}")
        self._toggle_region_dialog_fields()

    def _validated_region_dialog_values(self) -> Dict:
        values = {}
        for name in REGION_DETECTION_FIELDS:
            variable = self.region_dialog_vars[name]
            if name == "adaptive_threshold":
                values[name] = bool(variable.get())
                continue
            label, value_type, minimum, maximum = REGION_FIELD_META[name]
            raw_value = str(variable.get()).strip().replace(",", ".")
            try:
                numeric = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"«{label}»: введите число") from exc
            if not minimum <= numeric <= maximum:
                raise ValueError(
                    f"«{label}»: допустимо от {minimum:g} до {maximum:g}"
                )
            values[name] = (
                int(round(numeric)) if value_type is int else float(numeric)
            )
        if values["max_area"] < values["min_area"]:
            raise ValueError("Максимальная площадь должна быть не меньше минимальной")
        return values

    def _apply_region_settings_dialog(self, apply_to_all: bool = False):
        if self.region_dialog_enabled_var is None:
            return
        try:
            enabled = bool(self.region_dialog_enabled_var.get())
            values = self._validated_region_dialog_values() if enabled else None
            selected_id = self._region_dialog_selected_id()
            region_ids = (
                range(1, self._region_dialog_count() + 1)
                if apply_to_all
                else (selected_id,)
            )
            for region_id in region_ids:
                if values is None:
                    self.calibration.region_settings.pop(str(region_id), None)
                else:
                    self.calibration.region_settings[str(region_id)] = dict(values)
            if self.tracker:
                self._reset_processing_modules(announce=False)
            target = "всем областям" if apply_to_all else f"P{selected_id}"
            mode = "индивидуальные параметры применены" if enabled else "используются глобальные параметры"
            self.status_label.config(text=f"{target}: {mode}; поиск начат заново")
            self._load_region_settings_to_dialog()
        except Exception as exc:
            messagebox.showerror(
                "Индивидуальные настройки",
                str(exc),
                parent=self.region_settings_window,
            )

    def _disable_selected_region_settings(self):
        if self.region_dialog_enabled_var is None:
            return
        self.region_dialog_enabled_var.set(False)
        self._apply_region_settings_dialog()

    def _clear_all_region_settings(self):
        if not self.calibration.region_settings:
            self.status_label.config(text="Индивидуальные настройки уже отсутствуют")
            return
        if not messagebox.askyesno(
            "Сбросить настройки?",
            "Перевести все области на глобальные параметры обнаружения?",
            parent=self.region_settings_window,
        ):
            return
        self.calibration.region_settings.clear()
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self._load_region_settings_to_dialog()
        self.status_label.config(
            text="Все области переведены на глобальные параметры; поиск начат заново"
        )

    def _on_param_change(self, *args):
        if hasattr(self, "expected_var"):
            self.current_preset_name = "Пользовательский"
            if hasattr(self, "preset_var"):
                self.preset_var.set("Пользовательский")
            self._update_calibration_from_ui()
            self._schedule_processing_reset()

    def _on_expected_change(self, event=None):
        try:
            value = max(1, min(20, int(self.expected_var.get())))
        except (tk.TclError, ValueError):
            value = self.calibration.expected_reflectors
        self.expected_var.set(value)
        self.current_preset_name = "Пользовательский"
        if hasattr(self, "preset_var"):
            self.preset_var.set("Пользовательский")
        self._update_calibration_from_ui()
        self._update_region_status()
        self._refresh_region_dialog_selector()
        self._schedule_processing_reset()

    def _update_calibration_from_ui(self):
        try:
            self.calibration.min_area = int(self.min_area_var.get())
            self.calibration.max_area = int(self.max_area_var.get())
            self.calibration.circularity_threshold = float(self.circularity_var.get())
            self.calibration.brightness_threshold = int(self.brightness_var.get())
            self.calibration.contrast_threshold = int(self.contrast_var.get())
            self.calibration.blur_sigma = float(self.blur_var.get())
            self.calibration.expected_reflectors = max(1, int(self.expected_var.get()))
            self.calibration.adaptive_threshold = bool(self.adaptive_var.get())
            self.calibration.brightness_percentile = float(self.percentile_var.get())
            self.calibration.merge_radius = int(self.merge_radius_var.get())
            self.calibration.center_power = float(self.center_power_var.get())
            self.calibration.smoothing_alpha = float(self.smoothing_var.get())
            self.calibration.roi_expand_step = int(self.expand_step_var.get())
            self.calibration.roi_max_scale = float(self.max_scale_var.get())
            self.calibration.lost_hold_frames = int(self.hold_frames_var.get())
            self.calibration.max_jump = float(self.max_jump_var.get())
            if hasattr(self, "show_points_var"):
                self.calibration.show_points = bool(self.show_points_var.get())
                self.calibration.show_circles = bool(self.show_circles_var.get())
                self.calibration.show_frames = bool(self.show_frames_var.get())
                self.calibration.show_lines = bool(self.show_lines_var.get())
                self.calibration.show_distances = bool(
                    self.show_distances_var.get()
                )
                self.calibration.show_distance_changes = bool(
                    self.show_distance_changes_var.get()
                )
                self.calibration.close_shape = bool(self.close_shape_var.get())
            if hasattr(self, "ir_settle_var"):
                self.calibration.ir_settle_seconds = float(
                    self.ir_settle_var.get()
                )
                self.calibration.ir_flash_delta = int(
                    self.ir_flash_delta_var.get()
                )
                self.calibration.ir_search_scale = float(
                    self.ir_search_scale_var.get()
                )
                self.calibration.ir_lock_enabled = bool(
                    self.ir_lock_enabled_var.get()
                )
                self.calibration.ir_lock_radius = float(
                    self.ir_lock_radius_var.get()
                )
                self.calibration.hikvision_channel = max(
                    1, int(self.hik_channel_var.get())
                )
        except (tk.TclError, ValueError):
            return
        if hasattr(self, "region_status_label"):
            self._update_region_status()

    def _update_ui_from_calibration(self):
        self.min_area_var.set(self.calibration.min_area)
        self.max_area_var.set(self.calibration.max_area)
        self.circularity_var.set(self.calibration.circularity_threshold)
        self.brightness_var.set(self.calibration.brightness_threshold)
        self.contrast_var.set(self.calibration.contrast_threshold)
        self.blur_var.set(self.calibration.blur_sigma)
        self.expected_var.set(self.calibration.expected_reflectors)
        self.adaptive_var.set(self.calibration.adaptive_threshold)
        self.percentile_var.set(self.calibration.brightness_percentile)
        self.merge_radius_var.set(self.calibration.merge_radius)
        self.center_power_var.set(self.calibration.center_power)
        self.smoothing_var.set(self.calibration.smoothing_alpha)
        self.expand_step_var.set(self.calibration.roi_expand_step)
        self.max_scale_var.set(self.calibration.roi_max_scale)
        self.hold_frames_var.set(self.calibration.lost_hold_frames)
        self.max_jump_var.set(self.calibration.max_jump)
        if hasattr(self, "show_points_var"):
            self.show_points_var.set(self.calibration.show_points)
            self.show_circles_var.set(self.calibration.show_circles)
            self.show_frames_var.set(self.calibration.show_frames)
            self.show_lines_var.set(self.calibration.show_lines)
            self.show_distances_var.set(self.calibration.show_distances)
            self.show_distance_changes_var.set(
                self.calibration.show_distance_changes
            )
            self.close_shape_var.set(self.calibration.close_shape)
        if hasattr(self, "ir_settle_var"):
            self.ir_settle_var.set(self.calibration.ir_settle_seconds)
            self.ir_flash_delta_var.set(self.calibration.ir_flash_delta)
            self.ir_search_scale_var.set(self.calibration.ir_search_scale)
            self.ir_lock_enabled_var.set(self.calibration.ir_lock_enabled)
            self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)
            self.hik_channel_var.set(self.calibration.hikvision_channel)
        self._update_region_status()
        self._refresh_region_dialog_selector()

    def _update_region_status(self):
        if not hasattr(self, "region_status_label"):
            return
        count = len(self.calibration.peak_regions)
        expected = int(self.calibration.expected_reflectors)
        color = "green" if count == expected else "#b05a00"
        self.region_status_label.config(
            text=f"Области: {count}/{expected}", foreground=color
        )

    def _current_preset_parameters(self) -> Dict:
        self._update_calibration_from_ui()
        return {
            field: getattr(self.calibration, field)
            for field in PRESET_FIELDS
        }

    def _apply_preset_parameters(self, parameters: Dict, name: str):
        if not isinstance(parameters, dict):
            raise ValueError("В файле отсутствует словарь параметров пресета")
        applied = 0
        for field in PRESET_FIELDS:
            if field not in parameters:
                continue
            value = parameters[field]
            reference = BUILTIN_PRESETS["Базовый"][field]
            if isinstance(reference, bool):
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "да")
                else:
                    value = bool(value)
            elif isinstance(reference, int):
                value = int(round(float(value)))
            else:
                value = float(value)
            setattr(self.calibration, field, value)
            applied += 1
        if applied == 0:
            raise ValueError("В файле не найдено параметров обнаружения и сопровождения")

        self.calibration.expected_reflectors = max(
            1, min(20, int(self.calibration.expected_reflectors))
        )
        self.current_preset_name = name or "Пользовательский"
        if hasattr(self, "preset_var"):
            self.preset_var.set(self.current_preset_name)
        self._update_ui_from_calibration()
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self._update_region_status()
        self.status_label.config(
            text=(
                f"Применен пресет «{self.current_preset_name}»; "
                "детектор и трекер запущены заново"
            )
        )

    def apply_selected_preset(self):
        name = self.preset_var.get()
        parameters = BUILTIN_PRESETS.get(name) or self.user_presets.get(name)
        if parameters is None:
            messagebox.showerror("Пресет", "Выберите сохранённый пресет из списка")
            return
        try:
            self._apply_preset_parameters(parameters, name)
        except Exception as exc:
            messagebox.showerror("Ошибка пресета", str(exc))

    def create_user_preset(self):
        name = simpledialog.askstring(
            "Новый пользовательский пресет",
            "Введите название пресета:",
            parent=self.root,
        )
        if name is None:
            return
        name = " ".join(name.strip().split())
        if not name:
            messagebox.showerror("Пресет", "Название пресета не может быть пустым")
            return
        if len(name) > 80:
            messagebox.showerror("Пресет", "Название не должно превышать 80 символов")
            return
        if name in BUILTIN_PRESETS:
            messagebox.showerror(
                "Пресет",
                "Это название занято встроенным пресетом. Выберите другое название.",
            )
            return
        if name in self.user_presets and not messagebox.askyesno(
            "Перезаписать пресет?",
            f"Пользовательский пресет «{name}» уже существует. Перезаписать его?",
        ):
            return
        try:
            self.user_presets[name] = self._current_preset_parameters()
            self._save_user_preset_store()
            self.current_preset_name = name
            self._refresh_preset_combo(selected=name)
            self.status_label.config(
                text=f"Создан пользовательский пресет «{name}»"
            )
        except Exception as exc:
            messagebox.showerror("Ошибка создания пресета", str(exc))

    def delete_user_preset(self):
        name = self.preset_var.get()
        if name in BUILTIN_PRESETS:
            messagebox.showinfo("Пресет", "Встроенные пресеты удалять нельзя")
            return
        if name not in self.user_presets:
            messagebox.showerror("Пресет", "Выберите пользовательский пресет")
            return
        if not messagebox.askyesno(
            "Удалить пресет?",
            f"Удалить пользовательский пресет «{name}»?",
        ):
            return
        try:
            del self.user_presets[name]
            self._save_user_preset_store()
            if self.current_preset_name == name:
                self.current_preset_name = "Пользовательский"
            self._refresh_preset_combo(selected="Базовый")
            self.status_label.config(
                text=f"Пользовательский пресет «{name}» удалён; текущие параметры сохранены"
            )
        except Exception as exc:
            messagebox.showerror("Ошибка удаления пресета", str(exc))

    def save_preset_file(self):
        parameters = self._current_preset_parameters()
        default_name = self.current_preset_name or "Пользовательский"
        filename = filedialog.asksaveasfilename(
            title="Сохранить пресет отдельно",
            defaultextension=".json",
            initialfile=f"preset_{default_name.replace(' ', '_')}.json",
            filetypes=[("Пресет JSON", "*.json")],
        )
        if not filename:
            return
        data = {
            "format": "reflector-tracker-preset",
            "version": 1,
            "name": default_name,
            "parameters": parameters,
        }
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            self.status_label.config(
                text=f"Пресет отдельно сохранен: {os.path.basename(filename)}"
            )
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения пресета", str(exc))

    def load_preset_file(self):
        filename = filedialog.askopenfilename(
            title="Загрузить отдельный пресет",
            filetypes=[("Пресет JSON", "*.json")],
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as file:
                data = json.load(file)
            parameters = data.get("parameters", data)
            name = " ".join(str(data.get("name") or Path(filename).stem).strip().split())
            if not name:
                raise ValueError("В импортируемом файле отсутствует название пресета")
            if name in BUILTIN_PRESETS:
                name = f"{name} (импорт)"
            if name in self.user_presets and not messagebox.askyesno(
                "Перезаписать пресет?",
                f"Пользовательский пресет «{name}» уже существует. Перезаписать его?",
            ):
                return
            self._apply_preset_parameters(parameters, name)
            self.user_presets[name] = self._current_preset_parameters()
            self._save_user_preset_store()
            self._refresh_preset_combo(selected=name)
            self.status_label.config(
                text=f"Импортирован пользовательский пресет «{name}»"
            )
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки пресета", str(exc))

    def toggle_window_recording(self):
        if self.window_recorder.is_recording or self._record_output_path:
            self.stop_window_recording()
        else:
            self.start_window_recording()

    def start_window_recording(self):
        if self.current_frame is None or not (self.is_running or self.is_calibrating):
            messagebox.showwarning(
                "Запись",
                "Сначала запустите отслеживание или калибровку и дождитесь видеокадра.",
            )
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = filedialog.asksaveasfilename(
            title="Куда сохранить запись окна",
            defaultextension=".mp4",
            initialfile=f"reflector_monitor_{timestamp}.mp4",
            filetypes=[("Видео MP4", "*.mp4")],
        )
        if not filename:
            return

        try:
            fps = max(0.5, float(self.record_fps_var.get()))
            self.root.update_idletasks()
            width = max(2, self.root.winfo_width())
            height = max(2, self.root.winfo_height())
            self.window_recorder.start(filename, width, height, fps)

            output = Path(filename)
            state_path = output.with_name(f"{output.stem}_states.csv")
            settings_path = output.with_name(f"{output.stem}_settings.json")
            self._record_log_file = open(
                state_path, "w", newline="", encoding="utf-8-sig"
            )
            self._record_log_writer = csv.writer(self._record_log_file, delimiter=";")
            self._record_log_writer.writerow(
                [
                    "timestamp",
                    "elapsed_s",
                    "record_frame",
                    "peak_id",
                    "state",
                    "x_px",
                    "y_px",
                    "confidence",
                    "missed_frames",
                    "search_scale",
                    "area_px",
                    "threshold",
                    "preset",
                ]
            )
            settings_snapshot = {
                "format": "reflector-tracker-recording-settings",
                "version": 1,
                "recording_started": datetime.now().isoformat(timespec="seconds"),
                "recording_fps": fps,
                "preset": self.current_preset_name,
                "source_type": "rtsp" if self.use_rtsp else "local_camera",
                "calibration": self.calibration.to_dict(),
            }
            with open(settings_path, "w", encoding="utf-8") as file:
                json.dump(settings_snapshot, file, indent=4, ensure_ascii=False)

            self._record_output_path = filename
            self._record_start_time = time.time()
            self._record_frame_index = 0
            self.record_btn.config(text="⏹ Остановить запись")
            self.record_status_label.config(text="● REC 00:00:00", foreground="red")
            self.status_label.config(
                text="Запись окна начата. Не сворачивайте и не перекрывайте приложение."
            )
            self._capture_recording_frame()
        except Exception as exc:
            if self._record_log_file is not None:
                self._record_log_file.close()
            self._record_log_file = None
            self._record_log_writer = None
            self.window_recorder.stop()
            self._record_output_path = None
            messagebox.showerror("Ошибка запуска записи", str(exc))

    def _write_record_state(self, now: float):
        if self._record_log_writer is None:
            return
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        elapsed = now - self._record_start_time
        tracks = self.tracker.tracks if self.tracker else []
        if not tracks:
            self._record_log_writer.writerow(
                [
                    timestamp,
                    f"{elapsed:.3f}",
                    self._record_frame_index,
                    "",
                    "NO_TRACKS",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    self.current_preset_name,
                ]
            )
        else:
            for track in tracks:
                if track.measured_this_frame:
                    state = "LOCK"
                elif track.is_active:
                    state = "HOLD"
                elif track.has_measurement:
                    state = "LOST"
                else:
                    state = "SEARCH"
                x_value = f"{track.position[0]:.4f}" if track.has_measurement else ""
                y_value = f"{track.position[1]:.4f}" if track.has_measurement else ""
                self._record_log_writer.writerow(
                    [
                        timestamp,
                        f"{elapsed:.3f}",
                        self._record_frame_index,
                        track.id,
                        state,
                        x_value,
                        y_value,
                        f"{track.confidence:.4f}",
                        track.missed_frames,
                        f"{track.search_scale:.3f}",
                        f"{track.detection_area:.2f}",
                        f"{track.threshold:.2f}",
                        self.current_preset_name,
                    ]
                )
        self._record_log_file.flush()

    def _capture_recording_frame(self):
        self._record_capture_job = None
        if not self._record_output_path:
            return
        if not self.window_recorder.is_recording:
            error = self.window_recorder.error_message or "FFmpeg остановил запись"
            self.stop_window_recording(silent=True)
            messagebox.showerror("Ошибка записи", error)
            return
        try:
            self.root.update_idletasks()
            x1 = self.root.winfo_rootx()
            y1 = self.root.winfo_rooty()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            screenshot = ImageGrab.grab(
                bbox=(x1, y1, x1 + width, y1 + height),
                all_screens=True,
            ).convert("RGB")
            target_size = (self.window_recorder.width, self.window_recorder.height)
            if screenshot.size != target_size:
                resampling = getattr(Image, "Resampling", Image)
                screenshot = screenshot.resize(target_size, resampling.BILINEAR)
            self.window_recorder.submit(np.asarray(screenshot, dtype=np.uint8))

            now = time.time()
            self._record_frame_index += 1
            self._write_record_state(now)
            elapsed = max(0, int(now - self._record_start_time))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            dropped = self.window_recorder.frames_dropped
            dropped_text = f"; пропущено {dropped}" if dropped else ""
            self.record_status_label.config(
                text=f"● REC {hours:02d}:{minutes:02d}:{seconds:02d}{dropped_text}",
                foreground="red",
            )
            interval = max(1, int(round(1000.0 / self.window_recorder.fps)))
            self._record_capture_job = self.root.after(
                interval, self._capture_recording_frame
            )
        except Exception as exc:
            self.stop_window_recording(silent=True)
            messagebox.showerror("Ошибка захвата окна", str(exc))

    def stop_window_recording(self, silent: bool = False):
        output_path = self._record_output_path
        if self._record_capture_job is not None:
            try:
                self.root.after_cancel(self._record_capture_job)
            except tk.TclError:
                pass
            self._record_capture_job = None
        self._record_output_path = None
        self.window_recorder.stop()
        if self._record_log_file is not None:
            try:
                self._record_log_file.flush()
                self._record_log_file.close()
            except OSError:
                pass
        self._record_log_file = None
        self._record_log_writer = None
        if hasattr(self, "record_btn"):
            self.record_btn.config(text="⏺ Начать запись окна")
            self.record_status_label.config(text="Запись выключена", foreground="gray")
        if not silent and output_path:
            error = self.window_recorder.error_message
            if error:
                messagebox.showerror("Ошибка записи", error)
            else:
                self.status_label.config(
                    text=f"Запись сохранена: {os.path.basename(output_path)}"
                )

    def save_calibration(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if filename:
            self._update_calibration_from_ui()
            self.calibration.save(filename)
            self.status_label.config(
                text=f"Настройки сохранены: {os.path.basename(filename)}"
            )

    def load_calibration(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            self.calibration.load(filename)
            self.current_preset_name = "Полные настройки"
            self.preset_var.set(self.current_preset_name)
            self._update_ui_from_calibration()
            if self.tracker:
                self._reset_processing_modules(announce=False)
            self.status_label.config(
                text=f"Настройки загружены: {os.path.basename(filename)}"
            )
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))

    def _connect_rtsp(self) -> bool:
        self._disconnect_all()
        url = self.rtsp_url_var.get().strip()
        if not url:
            messagebox.showerror("Ошибка", "Введите RTSP URL")
            return False
        self.status_label.config(text="Подключение к RTSP через FFmpeg...")
        self.root.update_idletasks()
        self.rtsp_camera = RTSPCamera(url)
        if self.rtsp_camera.connect():
            self.connection_status.config(text="RTSP подключен", foreground="green")
            return True
        self.rtsp_camera = None
        self.connection_status.config(text="Ошибка RTSP", foreground="red")
        messagebox.showerror(
            "Ошибка подключения",
            "Не удалось подключиться к RTSP. Проверьте URL, сеть, логин, пароль и FFmpeg.",
        )
        return False

    def _connect_local(self) -> bool:
        self._disconnect_all()
        try:
            camera_id = int(self.local_camera_var.get())
            self.local_cap = cv2.VideoCapture(camera_id)
            if self.local_cap.isOpened():
                self.connection_status.config(text="Камера подключена", foreground="green")
                return True
        except Exception as exc:
            logger.error("Ошибка локальной камеры: %s", exc)
        self.local_cap = None
        self.connection_status.config(text="Ошибка камеры", foreground="red")
        messagebox.showerror("Ошибка", "Не удалось открыть локальную камеру")
        return False

    def _connect_selected_source(self) -> bool:
        return self._connect_rtsp() if self.use_rtsp else self._connect_local()

    def _disconnect_all(self):
        if self.rtsp_camera:
            self.rtsp_camera.stop()
            self.rtsp_camera = None
        if self.local_cap is not None:
            self.local_cap.release()
            self.local_cap = None
        if hasattr(self, "connection_status"):
            self.connection_status.config(text="Не подключено", foreground="red")

    def toggle_detection(self):
        if self.is_running:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self):
        if self.rtsp_camera is None and self.local_cap is None:
            if not self._connect_selected_source():
                return
        self._update_calibration_from_ui()
        self.tracker = ReflectorTracker(self.calibration)
        self.tracker.detector.background = self.background_frame
        self.is_running = True
        self.start_btn.config(text="⏸ Остановить отслеживание")
        count = len(self.calibration.peak_regions)
        if count != self.calibration.expected_reflectors:
            self.status_label.config(
                text=(
                    "Отслеживание запущено. Задайте по одной области для каждого "
                    f"пика: сейчас {count}/{self.calibration.expected_reflectors}."
                )
            )
        else:
            self.status_label.config(text="Отслеживание запущено")

    def _stop_detection(self):
        self.is_running = False
        self.start_btn.config(text="▶ Запуск отслеживания")
        self.status_label.config(text="Отслеживание остановлено")

    def toggle_calibration(self):
        if not self.is_calibrating:
            if self.rtsp_camera is None and self.local_cap is None:
                if not self._connect_selected_source():
                    return
            self._update_calibration_from_ui()
            if self.tracker is None:
                self.tracker = ReflectorTracker(self.calibration)
                self.tracker.detector.background = self.background_frame
            self.is_calibrating = True
            self.calibrate_btn.config(text="⚙ Калибровка (ВКЛ)")
            self.status_label.config(
                text="Калибровка: задайте области и настройте параметры"
            )
        else:
            self.is_calibrating = False
            self.calibrate_btn.config(text="⚙ Режим калибровки")
            self.status_label.config(text="Калибровка завершена")

    def capture_background(self):
        if self.current_frame is None:
            self.status_label.config(text="Нет кадра для захвата фона")
            return
        self.background_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self.status_label.config(
            text="Фон захвачен; детектор и трекер запущены заново"
        )

    def select_peak_regions(self):
        if self.current_frame is None:
            self.status_label.config(
                text="Сначала включите калибровку или отслеживание и дождитесь кадра"
            )
            return
        self._update_calibration_from_ui()
        expected = int(self.calibration.expected_reflectors)
        frame = self.current_frame.copy()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        scale = min(
            1.0,
            screen_w * 0.82 / frame.shape[1],
            screen_h * 0.78 / frame.shape[0],
        )
        display = cv2.resize(
            frame,
            (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        )
        selected: List[Rect] = []
        self.status_label.config(
            text="Поочередно выделите область каждого пика; Enter — принять, Esc — отменить"
        )
        for index in range(expected):
            preview = display.copy()
            for previous_index, region in enumerate(selected):
                rx, ry, rw, rh = region
                scaled_rect = (
                    int(rx * scale),
                    int(ry * scale),
                    int(rw * scale),
                    int(rh * scale),
                )
                sx, sy, sw, sh = scaled_rect
                cv2.rectangle(preview, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"P{previous_index + 1}",
                    (sx, max(16, sy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                preview,
                f"Select peak P{index + 1}/{expected}; ENTER=OK, ESC=cancel",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
            )
            window_name = f"Peak P{index + 1} of {expected}"
            roi = cv2.selectROI(window_name, preview, False, False)
            cv2.destroyWindow(window_name)
            if roi[2] <= 0 or roi[3] <= 0:
                cv2.destroyAllWindows()
                self.status_label.config(text="Выбор областей отменен; старые области сохранены")
                return
            selected.append(
                (
                    int(round(roi[0] / scale)),
                    int(round(roi[1] / scale)),
                    int(round(roi[2] / scale)),
                    int(round(roi[3] / scale)),
                )
            )

        self.calibration.peak_regions = selected
        # Ручной повторный выбор областей отменяет старую ИК-привязку.
        self.calibration.ir_confirmed_centers.clear()
        xs = [region[0] for region in selected]
        ys = [region[1] for region in selected]
        x2s = [region[0] + region[2] for region in selected]
        y2s = [region[1] + region[3] for region in selected]
        self.calibration.roi_region = (
            min(xs),
            min(ys),
            max(x2s) - min(xs),
            max(y2s) - min(ys),
        )
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self._update_region_status()
        self.status_label.config(
            text=(
                f"Задано областей пиков: {len(selected)}; "
                "детектор и трекер запущены заново"
            )
        )

    def clear_peak_regions(self):
        self.calibration.peak_regions = []
        self.calibration.roi_region = None
        self.calibration.ir_confirmed_centers.clear()
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self._update_region_status()
        self.status_label.config(
            text="Области пиков очищены; детектор и трекер запущены заново"
        )

    def _get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.use_rtsp and self.rtsp_camera:
            return self.rtsp_camera.read_frame()
        if self.local_cap is not None:
            ok, frame = self.local_cap.read()
            return (True, frame) if ok and frame is not None else (False, None)
        return False, None

    def _update_display(self):
        try:
            connected = self.rtsp_camera is not None or self.local_cap is not None
            if connected and (self.is_running or self.is_calibrating):
                ok, frame = self._get_frame()
                if ok and frame is not None:
                    self.current_frame = frame.copy()
                    if self.tracker is None:
                        self.tracker = ReflectorTracker(self.calibration)
                        self.tracker.detector.background = self.background_frame
                    if self.show_preprocessing:
                        processed = self.tracker.detector.preprocess_frame(frame)
                        display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    else:
                        display, _ = self.tracker.process_frame(frame)
                        if self.is_calibrating:
                            cv2.putText(
                                display,
                                "CALIBRATION MODE",
                                (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.70,
                                (0, 255, 255),
                                2,
                            )
                    if (
                        self._ir_preview_overlay is not None
                        and time.time() < self._ir_preview_until
                    ):
                        display = self._ir_preview_overlay.copy()
                    elif self._ir_preview_overlay is not None:
                        self._ir_preview_overlay = None
                    self._show_frame(display)
                    now = time.time()
                    elapsed = now - self._last_frame_time
                    if elapsed > 0:
                        self.fps_label.config(text=f"FPS: {1.0 / elapsed:.1f}")
                    self._last_frame_time = now
                else:
                    self.fps_label.config(text="FPS: --")
            else:
                self._show_placeholder()
        except Exception as exc:
            logger.exception("Ошибка обработки кадра: %s", exc)
            self.status_label.config(text=f"Ошибка обработки: {exc}")
        self.root.after(30, self._update_display)

    def _show_placeholder(self):
        self.main_canvas.delete("all")
        width = self.main_canvas.winfo_width()
        height = self.main_canvas.winfo_height()
        if width > 1 and height > 1:
            self.main_canvas.create_text(
                width // 2,
                height // 2,
                text="Выберите источник и включите калибровку или отслеживание",
                fill="gray",
                font=("Arial", 16),
                width=max(300, width - 80),
            )

    def _show_frame(self, frame: np.ndarray):
        canvas_w = self.main_canvas.winfo_width()
        canvas_h = self.main_canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        frame_h, frame_w = frame.shape[:2]
        self._clamp_view_center()
        crop_w = max(2, min(frame_w, int(round(frame_w / self.view_zoom))))
        crop_h = max(2, min(frame_h, int(round(frame_h / self.view_zoom))))
        center_x = int(round(self.view_center[0] * frame_w))
        center_y = int(round(self.view_center[1] * frame_h))
        x1 = max(0, min(frame_w - crop_w, center_x - crop_w // 2))
        y1 = max(0, min(frame_h - crop_h, center_y - crop_h // 2))
        view = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]

        scale = min(canvas_w / view.shape[1], canvas_h / view.shape[0])
        new_w = max(1, int(round(view.shape[1] * scale)))
        new_h = max(1, int(round(view.shape[0] * scale)))
        self._view_render_size = (new_w, new_h)
        resized = cv2.resize(view, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.main_canvas.delete("all")
        self.main_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER
        )
        if self.view_zoom > 1.0:
            self.main_canvas.create_text(
                canvas_w - 12,
                12,
                text=f"{self.view_zoom:.2f}×  •  перетаскивание мышью",
                fill="white",
                font=("Arial", 10, "bold"),
                anchor=tk.NE,
            )
        self.main_canvas.image = photo

    def _on_closing(self):
        self.is_running = False
        self.is_calibrating = False
        self._ir_scan_token += 1
        self._ir_scan_running = False
        if self._record_output_path or self.window_recorder.is_recording:
            self.stop_window_recording(silent=True)
        self._disconnect_all()
        self.root.destroy()

    def run(self):
        logger.info("OpenCV: %s", cv2.__version__)
        self.root.mainloop()


def main():
    ReflectorApp().run()


if __name__ == "__main__":
    main()
