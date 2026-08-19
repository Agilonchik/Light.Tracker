"""Строгий детектор признака «чёрный Day -> яркий Night».

Важные ограничения:
1. Каждая ROI Pn обрабатывается отдельно.
2. Пиксели за границей ROI никогда не участвуют в сегментации.
3. Используется только положительная разность Night - Day.
4. Кандидат обязан повториться минимум в двух циклах Day -> Night.
5. Центр вычисляется по яркому Night-пятну, а Day используется только для
   подтверждения того, что в этом месте было компактное чёрное ядро.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass
class DetectorSettings:
    day_black_max: int = 85
    night_bright_min: int = 155
    min_positive_gain: int = 65
    min_area: int = 3
    max_area: int = 1400
    blur_sigma: float = 0.55
    close_radius: int = 1
    day_match_radius: int = 5
    repeat_max_distance: float = 12.0
    center_power: float = 2.2
    minimum_score: float = 0.42
    minimum_diamond_score: float = 0.0
    registration_enabled: bool = True
    registration_max_shift: float = 35.0

    def validate(self) -> None:
        if not 0 <= self.day_black_max <= 255:
            raise ValueError("day_black_max должен быть в диапазоне 0...255")
        if not 0 <= self.night_bright_min <= 255:
            raise ValueError("night_bright_min должен быть в диапазоне 0...255")
        if not 0 <= self.min_positive_gain <= 255:
            raise ValueError("min_positive_gain должен быть в диапазоне 0...255")
        if self.min_area < 1 or self.max_area < self.min_area:
            raise ValueError("Некорректный диапазон площади")
        if self.center_power < 1.0:
            raise ValueError("center_power должен быть не меньше 1")


@dataclass
class DetectionResult:
    region_id: int
    roi: Rect
    found: bool
    center: Optional[Point] = None
    bbox: Optional[Rect] = None
    contour: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0
    reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = {
            "region_id": self.region_id,
            "roi": list(self.roi),
            "found": self.found,
            "center": list(self.center) if self.center is not None else None,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "score": self.score,
            "reason": self.reason,
            "metrics": self.metrics,
        }
        return data


@dataclass
class RegionDiagnostics:
    response_cycle_1: np.ndarray
    response_cycle_2: np.ndarray
    mask_cycle_1: np.ndarray
    mask_cycle_2: np.ndarray


@dataclass
class DetectionBatch:
    results: List[DetectionResult]
    reference_night: np.ndarray
    day_medians: List[np.ndarray]
    night_medians: List[np.ndarray]
    registrations: Dict[str, Tuple[float, float, float]]
    diagnostics: Dict[int, RegionDiagnostics]
    settings: DetectorSettings

    def annotated_frame(self) -> np.ndarray:
        frame = cv2.cvtColor(self.reference_night, cv2.COLOR_GRAY2BGR)
        for result in self.results:
            x, y, w, h = result.roi
            color = (0, 220, 0) if result.found else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"P{result.region_id}"
            if result.found and result.center is not None:
                label += f" FOUND S:{result.score:.2f}"
                if result.contour is not None and len(result.contour):
                    cv2.drawContours(frame, [result.contour], -1, (0, 210, 255), 1)
                draw_cross(frame, result.center, (255, 0, 255), 12, 2)
            else:
                label += f" NOT FOUND: {result.reason}"
            cv2.putText(
                frame,
                label,
                (x + 2, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def save(self, directory: Path | str) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(self.day_medians, start=1):
            cv2.imwrite(str(path / f"day_cycle_{index}.png"), image)
        for index, image in enumerate(self.night_medians, start=1):
            cv2.imwrite(str(path / f"night_cycle_{index}.png"), image)
        cv2.imwrite(str(path / "result_annotated.png"), self.annotated_frame())
        for region_id, diagnostic in self.diagnostics.items():
            for name, image in (
                ("response_cycle_1", diagnostic.response_cycle_1),
                ("response_cycle_2", diagnostic.response_cycle_2),
                ("mask_cycle_1", diagnostic.mask_cycle_1),
                ("mask_cycle_2", diagnostic.mask_cycle_2),
            ):
                output = image
                if output.dtype != np.uint8:
                    maximum = max(1.0, float(np.max(output)))
                    output = np.clip(output / maximum * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(path / f"P{region_id}_{name}.png"), output)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": asdict(self.settings),
            "registrations": {
                key: [float(value) for value in values]
                for key, values in self.registrations.items()
            },
            "results": [result.to_dict() for result in self.results],
        }
        with open(path / "result.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return path


@dataclass
class _Candidate:
    center: Point
    bbox: Rect
    contour_local: np.ndarray
    score: float
    area: int
    day_level: float
    night_level: float
    gain: float
    diamond_score: float


def draw_cross(
    image: np.ndarray,
    center: Point,
    color: Tuple[int, int, int],
    size: int = 10,
    thickness: int = 2,
) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _clip_rect(rect: Rect, shape: Tuple[int, ...]) -> Optional[Rect]:
    height, width = shape[:2]
    x, y, w, h = [int(round(value)) for value in rect]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _median_image(samples: Sequence[np.ndarray]) -> np.ndarray:
    if not samples:
        raise ValueError("Пустой набор кадров")
    gray = [_gray(sample) for sample in samples]
    shapes = {image.shape for image in gray}
    if len(shapes) != 1:
        raise ValueError("Размер кадров внутри одного состояния различается")
    return np.median(np.stack(gray), axis=0).astype(np.uint8)


def _edge_image(image: np.ndarray) -> np.ndarray:
    source = image.astype(np.float32)
    low, high = np.percentile(source, (3.0, 97.0))
    normalized = np.clip((source - low) / max(1.0, high - low), 0.0, 1.0)
    normalized = cv2.GaussianBlur(normalized, (0, 0), 1.1)
    dx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    return np.clip(
        magnitude / max(1e-6, float(np.percentile(magnitude, 98.0))),
        0.0,
        1.0,
    ).astype(np.float32)


def _align_translation(
    image: np.ndarray,
    reference: np.ndarray,
    maximum_shift: float,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Совмещает кадр с первым Night только геометрически, без поиска точек."""
    try:
        height, width = reference.shape
        scale = min(1.0, 900.0 / max(height, width))
        if scale < 1.0:
            size = (max(96, int(width * scale)), max(96, int(height * scale)))
            source_small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            reference_small = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        else:
            source_small, reference_small = image, reference
        warp = np.eye(2, 3, dtype=np.float32)
        correlation, warp = cv2.findTransformECC(
            _edge_image(reference_small),
            _edge_image(source_small),
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )
        shift_x = float(warp[0, 2] / scale)
        shift_y = float(warp[1, 2] / scale)
        if (
            not np.isfinite(correlation)
            or correlation < 0.12
            or abs(shift_x) > maximum_shift
            or abs(shift_y) > maximum_shift
        ):
            return image.copy(), (0.0, 0.0, 0.0)
        full_warp = warp.copy()
        full_warp[0, 2] = shift_x
        full_warp[1, 2] = shift_y
        aligned = cv2.warpAffine(
            image,
            full_warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned, (shift_x, shift_y, float(correlation))
    except (cv2.error, ValueError, FloatingPointError):
        return image.copy(), (0.0, 0.0, 0.0)


def _diamond_score(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = max(1.0, cv2.contourArea(contour))
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
    hull_area = max(area, cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area
    vertex_score = max(0.0, 1.0 - abs(len(polygon) - 4) / 5.0)
    return float(np.clip(0.55 * solidity + 0.45 * vertex_score, 0.0, 1.0))


def _stable_bright_center(
    component: np.ndarray,
    response: np.ndarray,
    night: np.ndarray,
    power: float,
) -> Point:
    values = response[component]
    if values.size == 0:
        raise ValueError("Пустая компонента")
    maximum = float(np.max(values))
    centers: List[Tuple[float, float, float]] = []
    yy, xx = np.indices(component.shape, dtype=np.float64)
    base = np.maximum(response, 0.0) * np.maximum(night.astype(np.float32), 1.0) / 255.0
    for fraction in (0.40, 0.58, 0.74, 0.86):
        level_mask = component & (response >= maximum * fraction)
        if np.count_nonzero(level_mask) < 1:
            continue
        weights = np.where(level_mask, np.power(base + 1e-3, power), 0.0)
        total = float(np.sum(weights))
        if total <= 0:
            continue
        centers.append(
            (
                float(np.sum(xx * weights) / total),
                float(np.sum(yy * weights) / total),
                float(np.count_nonzero(level_mask)),
            )
        )
    if not centers:
        ys, xs = np.where(component)
        return float(np.mean(xs)), float(np.mean(ys))
    # Медиана центров нескольких уровней менее чувствительна к границе ореола.
    return (
        float(np.median([item[0] for item in centers])),
        float(np.median([item[1] for item in centers])),
    )


class DayNightReflectorDetector:
    """Не содержит трекера и не изменяет состояние камеры."""

    def __init__(self, settings: Optional[DetectorSettings] = None):
        self.settings = settings or DetectorSettings()
        self.settings.validate()

    def _cycle_candidates(
        self,
        day: np.ndarray,
        night: np.ndarray,
        roi: Rect,
    ) -> Tuple[List[_Candidate], np.ndarray, np.ndarray, str]:
        x, y, w, h = roi
        day_crop = day[y : y + h, x : x + w]
        night_crop = night[y : y + h, x : x + w]
        sigma = max(0.0, float(self.settings.blur_sigma))
        if sigma > 0:
            day_work = cv2.GaussianBlur(day_crop, (0, 0), sigma)
            night_work = cv2.GaussianBlur(night_crop, (0, 0), sigma)
        else:
            day_work, night_work = day_crop, night_crop

        radius = max(0, int(self.settings.day_match_radius))
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            # Минимум только ВНУТРИ ROI: borderValue=255 не втягивает чёрные
            # пиксели из соседней пользовательской области.
            day_local_min = cv2.erode(
                day_work,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
        else:
            day_local_min = day_work

        day_f = day_work.astype(np.float32)
        day_min_f = day_local_min.astype(np.float32)
        night_f = night_work.astype(np.float32)
        positive_gain = np.maximum(night_f - day_min_f, 0.0)
        exact_gain = np.maximum(night_f - day_f, 0.0)
        dark_weight = np.clip(
            (float(self.settings.day_black_max) + 12.0 - day_min_f) / 40.0,
            0.0,
            1.0,
        )
        bright_weight = np.clip(
            (night_f - float(self.settings.night_bright_min) + 20.0) / 80.0,
            0.0,
            1.0,
        )
        response = positive_gain * dark_weight * bright_weight
        raw_mask = (
            (day_local_min <= self.settings.day_black_max)
            & (night_work >= self.settings.night_bright_min)
            & (positive_gain >= self.settings.min_positive_gain)
        ).astype(np.uint8) * 255

        close_radius = max(0, int(self.settings.close_radius))
        if close_radius:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * close_radius + 1, 2 * close_radius + 1),
            )
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

        if not np.any(night_work >= self.settings.night_bright_min):
            reason = "NO_BRIGHT_NIGHT"
        elif not np.any(day_local_min <= self.settings.day_black_max):
            reason = "NO_BLACK_DAY"
        elif not np.any(positive_gain >= self.settings.min_positive_gain):
            reason = "NO_POSITIVE_GAIN"
        elif not np.any(raw_mask):
            reason = "NO_COMBINED_SIGNATURE"
        else:
            reason = "NO_VALID_COMPONENT"

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )
        candidates: List[_Candidate] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.settings.min_area or area > self.settings.max_area:
                continue
            component = labels == label
            component_exact_gain = exact_gain[component]
            # Хотя допускается небольшой Day/Night-сдвиг, часть Night-пятна
            # обязана действительно посветлеть в тех же пикселях.
            exact_fraction = float(
                np.mean(component_exact_gain >= self.settings.min_positive_gain * 0.35)
            )
            if exact_fraction < 0.18:
                continue
            center = _stable_bright_center(
                component,
                response,
                night_work,
                self.settings.center_power,
            )
            component_u8 = component.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            diamond = _diamond_score(component_u8)
            if diamond < self.settings.minimum_diamond_score:
                continue

            night_level = float(np.mean(night_f[component]))
            day_level = float(np.mean(day_min_f[component]))
            gain_level = float(np.mean(positive_gain[component]))
            transition_score = float(np.clip(gain_level / 190.0, 0.0, 1.0))
            darkness_score = float(
                np.clip((self.settings.day_black_max - day_level + 20.0) / 85.0, 0.0, 1.0)
            )
            brightness_score = float(
                np.clip((night_level - self.settings.night_bright_min + 35.0) / 120.0, 0.0, 1.0)
            )
            score = float(
                np.clip(
                    0.52 * transition_score
                    + 0.20 * darkness_score
                    + 0.18 * brightness_score
                    + 0.07 * diamond
                    + 0.03 * exact_fraction,
                    0.0,
                    1.0,
                )
            )
            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            candidates.append(
                _Candidate(
                    center=center,
                    bbox=bbox,
                    contour_local=contour,
                    score=score,
                    area=area,
                    day_level=day_level,
                    night_level=night_level,
                    gain=gain_level,
                    diamond_score=diamond,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates, response, raw_mask, reason

    def analyze(
        self,
        day_cycles: Sequence[Sequence[np.ndarray]],
        night_cycles: Sequence[Sequence[np.ndarray]],
        regions: Sequence[Rect],
    ) -> DetectionBatch:
        """Анализирует минимум два независимых цикла Day -> Night."""
        if len(day_cycles) < 2 or len(night_cycles) < 2:
            raise ValueError("Нужно минимум два полных цикла Day -> Night")
        day_medians = [_median_image(samples) for samples in day_cycles[:2]]
        night_medians = [_median_image(samples) for samples in night_cycles[:2]]
        shape_set = {image.shape for image in day_medians + night_medians}
        if len(shape_set) != 1:
            raise ValueError("Размер кадра изменился между Day и Night")

        reference = night_medians[0]
        registrations: Dict[str, Tuple[float, float, float]] = {
            "night_cycle_1": (0.0, 0.0, 1.0)
        }
        aligned_day: List[np.ndarray] = []
        aligned_night: List[np.ndarray] = [reference]
        for index, image in enumerate(day_medians):
            if self.settings.registration_enabled:
                aligned, transform = _align_translation(
                    image, reference, self.settings.registration_max_shift
                )
            else:
                aligned, transform = image.copy(), (0.0, 0.0, 0.0)
            aligned_day.append(aligned)
            registrations[f"day_cycle_{index + 1}"] = transform
        if self.settings.registration_enabled:
            aligned, transform = _align_translation(
                night_medians[1], reference, self.settings.registration_max_shift
            )
        else:
            aligned, transform = night_medians[1].copy(), (0.0, 0.0, 0.0)
        aligned_night.append(aligned)
        registrations["night_cycle_2"] = transform

        results: List[DetectionResult] = []
        diagnostics: Dict[int, RegionDiagnostics] = {}
        for region_id, original_roi in enumerate(regions, start=1):
            roi = _clip_rect(original_roi, reference.shape)
            if roi is None:
                results.append(
                    DetectionResult(region_id, original_roi, False, reason="INVALID_ROI")
                )
                continue
            first, response_1, mask_1, reason_1 = self._cycle_candidates(
                aligned_day[0], aligned_night[0], roi
            )
            second, response_2, mask_2, reason_2 = self._cycle_candidates(
                aligned_day[1], aligned_night[1], roi
            )
            diagnostics[region_id] = RegionDiagnostics(
                response_1, response_2, mask_1, mask_2
            )
            if not first or not second:
                reason = f"C1:{reason_1}" if not first else f"C2:{reason_2}"
                results.append(DetectionResult(region_id, roi, False, reason=reason))
                continue

            pairs = []
            for candidate_1 in first:
                for candidate_2 in second:
                    distance = float(
                        np.linalg.norm(
                            np.asarray(candidate_1.center) - np.asarray(candidate_2.center)
                        )
                    )
                    if distance > self.settings.repeat_max_distance:
                        continue
                    repeat_score = float(
                        np.clip(1.0 - distance / self.settings.repeat_max_distance, 0.0, 1.0)
                    )
                    total = (
                        0.42 * candidate_1.score
                        + 0.42 * candidate_2.score
                        + 0.16 * repeat_score
                    )
                    pairs.append((total, distance, candidate_1, candidate_2))
            if not pairs:
                results.append(
                    DetectionResult(region_id, roi, False, reason="NOT_REPEATABLE")
                )
                continue
            pairs.sort(key=lambda item: item[0], reverse=True)
            score, distance, candidate_1, candidate_2 = pairs[0]
            if score < self.settings.minimum_score:
                results.append(
                    DetectionResult(
                        region_id,
                        roi,
                        False,
                        score=float(score),
                        reason="LOW_SCORE",
                    )
                )
                continue

            x, y, _, _ = roi
            # Центр — среднее двух независимых Night-измерений. Day-центры
            # сюда принципиально не входят.
            local_center = (
                (candidate_1.center[0] + candidate_2.center[0]) / 2.0,
                (candidate_1.center[1] + candidate_2.center[1]) / 2.0,
            )
            center = (local_center[0] + x, local_center[1] + y)
            bx, by, bw, bh = candidate_2.bbox
            bbox = (bx + x, by + y, bw, bh)
            contour = candidate_2.contour_local.astype(np.int32).copy()
            contour[:, 0, 0] += x
            contour[:, 0, 1] += y
            results.append(
                DetectionResult(
                    region_id=region_id,
                    roi=roi,
                    found=True,
                    center=center,
                    bbox=bbox,
                    contour=contour,
                    score=float(score),
                    reason="OK",
                    metrics={
                        "repeat_distance_px": float(distance),
                        "day_level_max": float(
                            max(candidate_1.day_level, candidate_2.day_level)
                        ),
                        "night_level_min": float(
                            min(candidate_1.night_level, candidate_2.night_level)
                        ),
                        "positive_gain_min": float(
                            min(candidate_1.gain, candidate_2.gain)
                        ),
                        "area_cycle_1": float(candidate_1.area),
                        "area_cycle_2": float(candidate_2.area),
                        "diamond_score_min": float(
                            min(candidate_1.diamond_score, candidate_2.diamond_score)
                        ),
                    },
                )
            )

        return DetectionBatch(
            results=results,
            reference_night=reference,
            day_medians=aligned_day,
            night_medians=aligned_night,
            registrations=registrations,
            diagnostics=diagnostics,
            settings=self.settings,
        )
