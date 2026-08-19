"""Изолированные RTSP и ISAPI-компоненты для камер Hikvision.

Модуль ничего не знает о детекторе, интерфейсе и сопровождении точек. Его
единственная задача — получать свежие кадры и переключать IR-cut Day/Night.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shutil
import ssl
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import xml.etree.ElementTree as ET

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Не найден {name}. Установите FFmpeg и добавьте его в PATH."
        )
    return path


def credentials_from_rtsp(url: str) -> Tuple[str, str]:
    """Возвращает логин/пароль из RTSP URL без сохранения их на диск."""
    parsed = urllib_parse.urlsplit(url.strip())
    return (
        urllib_parse.unquote(parsed.username or ""),
        urllib_parse.unquote(parsed.password or ""),
    )


def isapi_url_from_rtsp(url: str) -> str:
    parsed = urllib_parse.urlsplit(url.strip())
    if not parsed.hostname:
        return ""
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}"


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
        self.ircut_endpoint: Optional[str] = None

        password_manager = urllib_request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(
            None, self.base_url, username or "admin", password or ""
        )
        handlers = [
            urllib_request.HTTPDigestAuthHandler(password_manager),
            urllib_request.HTTPBasicAuthHandler(password_manager),
        ]
        if parsed.scheme == "https":
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib_request.HTTPSHandler(context=context))
        self.opener = urllib_request.build_opener(*handlers)

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
                "User-Agent": "DayNightReflectorDetector/1.0",
            },
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Hikvision отклонила авторизацию. Проверьте логин, пароль "
                    "и разрешение ISAPI/CGI."
                ) from exc
            suffix = f": {detail[:180]}" if detail else ""
            raise RuntimeError(f"ISAPI HTTP {exc.code}{suffix}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Камера ISAPI недоступна: {exc.reason}") from exc

        if method == "PUT" and payload.strip():
            try:
                root = ET.fromstring(payload)
                code = self._find_xml_node(root, "statusCode")
                text = self._find_xml_node(root, "statusString")
                if code is not None and (code.text or "").strip() not in ("1", "OK"):
                    detail = (text.text or "").strip() if text is not None else "ошибка"
                    raise RuntimeError(f"Камера не применила режим: {detail}")
            except ET.ParseError:
                pass
        return payload

    def get_device_info(self) -> Dict[str, str]:
        payload = self._request("/ISAPI/System/deviceInfo")
        root = ET.fromstring(payload)
        result: Dict[str, str] = {}
        for name in ("model", "deviceName", "firmwareVersion", "serialNumber"):
            node = self._find_xml_node(root, name)
            if node is not None and node.text:
                result[name] = node.text.strip()
        return result

    def get_ircut(self) -> Tuple[str, bytes, str]:
        paths: List[str] = []
        if self.ircut_endpoint:
            paths.append(self.ircut_endpoint)
        paths.extend(
            [
                f"/ISAPI/Image/channels/{self.channel}/ircutFilter",
                f"/ISAPI/Image/channels/{self.channel}/IrcutFilter",
            ]
        )
        last_error: Optional[Exception] = None
        for path in dict.fromkeys(paths):
            try:
                payload = self._request(path)
                root = ET.fromstring(payload)
                node = self._find_xml_node(root, "IrcutFilterType")
                if node is None:
                    raise RuntimeError("В ответе камеры нет IrcutFilterType")
                self.ircut_endpoint = path
                return (node.text or "").strip().lower(), payload, path
            except Exception as exc:  # пробуем второй вариант регистра endpoint
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
            node = self._find_xml_node(root, "IrcutFilterType")
            if node is None:
                raise ET.ParseError("IrcutFilterType missing")
            node.text = mode
            if root.tag.startswith("{"):
                ET.register_namespace("", root.tag[1:].split("}", 1)[0])
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
    """Получает только самый свежий кадр RTSP через FFmpeg."""

    def __init__(self, url: str):
        self.url = url.strip()
        self.process: Optional[subprocess.Popen] = None
        self.frame_width = 0
        self.frame_height = 0
        self.frame_size = 0
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_sequence = 0
        self.closed = True
        self.is_connected = False
        self.last_error = ""
        self.condition = threading.Condition()
        self.reader_thread: Optional[threading.Thread] = None

    def _probe_frame_size(self) -> Tuple[int, int]:
        command = [
            require_executable("ffprobe"),
            "-v", "error",
            "-rtsp_transport", "tcp",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            self.url,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=12)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffprobe завершился с ошибкой")
        dimensions = result.stdout.strip().splitlines()[0]
        match = re.fullmatch(r"(\d+)x(\d+)", dimensions)
        if not match:
            raise RuntimeError(f"Неожиданный размер кадра: {dimensions}")
        return int(match.group(1)), int(match.group(2))

    def connect(self) -> None:
        if self.is_connected:
            return
        self.frame_width, self.frame_height = self._probe_frame_size()
        self.frame_size = self.frame_width * self.frame_height * 3
        command = [
            require_executable("ffmpeg"),
            "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.url,
            "-an", "-sn", "-dn",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.closed = False
        self.is_connected = True
        self.last_error = ""
        self.latest_frame = None
        self.latest_sequence = 0
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            name="day-night-rtsp-reader",
            daemon=True,
        )
        self.reader_thread.start()

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

    def _reader_loop(self) -> None:
        try:
            while not self.closed:
                if self.process is None or self.process.poll() is not None:
                    break
                payload = self._read_exactly(self.frame_size)
                if payload is None:
                    break
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                    self.frame_height, self.frame_width, 3
                ).copy()
                with self.condition:
                    self.latest_frame = frame
                    self.latest_sequence += 1
                    self.condition.notify_all()
        finally:
            with self.condition:
                if not self.closed:
                    self.last_error = "RTSP-поток FFmpeg завершился"
                self.is_connected = False
                self.condition.notify_all()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.condition:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def wait_next_frame(
        self,
        after_sequence: Optional[int] = None,
        timeout: float = 5.0,
    ) -> Tuple[np.ndarray, int]:
        deadline = time.monotonic() + max(0.1, timeout)
        with self.condition:
            baseline = self.latest_sequence if after_sequence is None else after_sequence
            while not self.closed and self.is_connected:
                if self.latest_frame is not None and self.latest_sequence > baseline:
                    return self.latest_frame.copy(), self.latest_sequence
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
        raise RuntimeError(self.last_error or "Не получен свежий RTSP-кадр")

    def collect_gray_frames(
        self,
        count: int,
        timeout_per_frame: float = 5.0,
    ) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        sequence = self.latest_sequence
        while len(frames) < max(1, int(count)):
            frame, sequence = self.wait_next_frame(sequence, timeout_per_frame)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return frames

    def stop(self) -> None:
        self.closed = True
        self.is_connected = False
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
        self.reader_thread = None
        self.latest_frame = None


@dataclass
class CaptureCycles:
    day_cycles: List[List[np.ndarray]]
    night_cycles: List[List[np.ndarray]]
    original_mode: str


class HikvisionCameraRig:
    """Связывает RTSP и ISAPI только на время Day/Night-съёмки."""

    def __init__(self, rtsp: RTSPCamera, isapi: HikvisionISAPI):
        self.rtsp = rtsp
        self.isapi = isapi

    def capture_day_night_cycles(
        self,
        cycles: int = 2,
        samples_per_state: int = 5,
        settle_seconds: float = 2.0,
        leave_in_night: bool = True,
        progress: Optional[Callable[[str], None]] = None,
    ) -> CaptureCycles:
        if not self.rtsp.is_connected:
            raise RuntimeError("RTSP-камера не подключена")
        cycles = max(2, int(cycles))
        samples_per_state = max(3, int(samples_per_state))
        settle_seconds = max(0.2, float(settle_seconds))
        original_mode, _, _ = self.isapi.get_ircut()
        day_cycles: List[List[np.ndarray]] = []
        night_cycles: List[List[np.ndarray]] = []

        try:
            for index in range(cycles):
                if progress:
                    progress(f"Цикл {index + 1}/{cycles}: переключение в Day")
                self.isapi.set_ircut("day")
                time.sleep(settle_seconds)
                day_cycles.append(
                    self.rtsp.collect_gray_frames(samples_per_state)
                )

                if progress:
                    progress(f"Цикл {index + 1}/{cycles}: переключение в Night")
                self.isapi.set_ircut("night")
                time.sleep(settle_seconds)
                night_cycles.append(
                    self.rtsp.collect_gray_frames(samples_per_state)
                )
        finally:
            target = "night" if leave_in_night else original_mode
            try:
                self.isapi.set_ircut(target)
            except Exception as exc:
                logger.warning("Не удалось восстановить режим камеры %s: %s", target, exc)

        return CaptureCycles(day_cycles, night_cycles, original_mode)
