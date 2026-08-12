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

# Базовый вид окна: только измеряемые точки и их смещения от стартового
# положения. Остальные слои пользователь включает независимо при необходимости.
BASE_DISPLAY_SETTINGS = {
    "show_points": True,
    "show_circles": False,
    "show_frames": False,
    "show_lines": False,
    "show_distances": False,
    "show_distance_changes": False,
    "show_displacements": True,
    "close_shape": False,
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


def make_regions_exclusive(
    proposed_regions: List[Rect],
    reference_regions: List[Rect],
    frame_shape: Tuple[int, ...],
) -> List[Rect]:
    """Обрезает расширенные области по границам исходных областей Pn.

    Если пользовательские области не пересекаются, их расширение больше не
    создает общую зону, в которой два трека могут выбрать один отражатель.
    Для диагонально расположенных областей выбирается наиболее выраженная ось
    разделения. Исходная пользовательская рамка всегда остается внутри своей
    территории.
    """
    if len(proposed_regions) != len(reference_regions):
        return [
            clipped
            for region in proposed_regions
            if (clipped := clip_rect(region, frame_shape)) is not None
        ]

    bounds = []
    for proposed, reference in zip(proposed_regions, reference_regions):
        clipped = clip_rect(proposed, frame_shape)
        if clipped is None:
            clipped = clip_rect(reference, frame_shape)
        if clipped is None:
            clipped = (0, 0, 3, 3)
        x, y, width, height = clipped
        bounds.append([float(x), float(y), float(x + width), float(y + height)])

    for first_index in range(len(reference_regions)):
        ax, ay, aw, ah = reference_regions[first_index]
        a_left, a_top = float(ax), float(ay)
        a_right, a_bottom = float(ax + aw), float(ay + ah)
        a_center_x = a_left + aw / 2.0
        a_center_y = a_top + ah / 2.0
        for second_index in range(first_index + 1, len(reference_regions)):
            bx, by, bw, bh = reference_regions[second_index]
            b_left, b_top = float(bx), float(by)
            b_right, b_bottom = float(bx + bw), float(by + bh)
            b_center_x = b_left + bw / 2.0
            b_center_y = b_top + bh / 2.0

            separators = []
            horizontal_score = abs(a_center_x - b_center_x) / max(
                1.0, (aw + bw) / 2.0
            )
            vertical_score = abs(a_center_y - b_center_y) / max(
                1.0, (ah + bh) / 2.0
            )
            if a_right <= b_left:
                separators.append(
                    (horizontal_score, "x", first_index, second_index,
                     (a_right + b_left) / 2.0)
                )
            elif b_right <= a_left:
                separators.append(
                    (horizontal_score, "x", second_index, first_index,
                     (b_right + a_left) / 2.0)
                )
            if a_bottom <= b_top:
                separators.append(
                    (vertical_score, "y", first_index, second_index,
                     (a_bottom + b_top) / 2.0)
                )
            elif b_bottom <= a_top:
                separators.append(
                    (vertical_score, "y", second_index, first_index,
                     (b_bottom + a_top) / 2.0)
                )

            if not separators:
                # Исходные области пересекаются: автоматически выбирать между
                # ними границу нельзя, поэтому оставляем их без изменения.
                continue
            _, axis, lower_index, upper_index, boundary = max(
                separators, key=lambda item: item[0]
            )
            boundary = float(round(boundary))
            if axis == "x":
                bounds[lower_index][2] = min(
                    bounds[lower_index][2], boundary
                )
                bounds[upper_index][0] = max(
                    bounds[upper_index][0], boundary
                )
            else:
                bounds[lower_index][3] = min(
                    bounds[lower_index][3], boundary
                )
                bounds[upper_index][1] = max(
                    bounds[upper_index][1], boundary
                )

    result = []
    for index, (left, top, right, bottom) in enumerate(bounds):
        candidate = clip_rect(
            (
                int(round(left)),
                int(round(top)),
                int(round(right - left)),
                int(round(bottom - top)),
            ),
            frame_shape,
        )
        if candidate is None:
            candidate = clip_rect(reference_regions[index], frame_shape)
        if candidate is not None:
            result.append(candidate)
    return result


def segment_enters_rect(
    start: Point,
    end: Point,
    rect: Rect,
) -> bool:
    """Проверяет вход отрезка во внутреннюю часть прямоугольника.

    Касание внешней границы не считается входом. Это позволяет двум соседним
    областям иметь общую границу, не блокируя корректный маршрут кандидата.
    """
    x, y, width, height = rect
    inset_x = min(0.5, max(0.0, (width - 1.0) / 2.0))
    inset_y = min(0.5, max(0.0, (height - 1.0) / 2.0))
    minimum_x = float(x) + inset_x
    maximum_x = float(x + width) - inset_x
    minimum_y = float(y) + inset_y
    maximum_y = float(y + height) - inset_y

    start_x, start_y = [float(value) for value in start]
    end_x, end_y = [float(value) for value in end]
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    enter = 0.0
    leave = 1.0
    for coordinate, delta, lower, upper in (
        (start_x, delta_x, minimum_x, maximum_x),
        (start_y, delta_y, minimum_y, maximum_y),
    ):
        if abs(delta) < 1e-12:
            if coordinate <= lower or coordinate >= upper:
                return False
            continue
        first = (lower - coordinate) / delta
        second = (upper - coordinate) / delta
        if first > second:
            first, second = second, first
        enter = max(enter, first)
        leave = min(leave, second)
        if enter > leave:
            return False

    # Начало маршрута находится в собственной области и она не проверяется.
    # Для чужой области нужен реальный участок внутри отрезка, а не касание
    # кандидатом границы в самой конечной точке.
    return bool(leave > 1e-6 and enter < 1.0 - 1e-6 and enter <= leave)


def route_blockers(
    start: Point,
    end: Point,
    reference_regions: List[Rect],
    source_region_index: int,
) -> List[int]:
    """Возвращает номера чужих базовых областей, пересеченных маршрутом."""
    return [
        index
        for index, region in enumerate(reference_regions)
        if index != source_region_index
        and segment_enters_rect(start, end, region)
    ]


def diamond_halo_score(component_mask: np.ndarray) -> float:
    """Оценивает сходство компонента с симметричным ромбовидным ореолом."""
    ys, xs = np.where(component_mask)
    if xs.size < 3:
        return 0.0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    crop = component_mask[y1 : y2 + 1, x1 : x2 + 1].astype(bool)
    height, width = crop.shape
    if width < 2 or height < 2:
        return 0.0

    aspect = max(width, height) / max(1.0, min(width, height))
    aspect_score = float(
        np.exp(-((np.log(max(aspect, 1.0)) / 0.55) ** 2))
    )
    horizontal_symmetry = 1.0 - float(
        np.mean(np.logical_xor(crop, np.fliplr(crop)))
    )
    vertical_symmetry = 1.0 - float(
        np.mean(np.logical_xor(crop, np.flipud(crop)))
    )
    symmetry = float(
        np.clip(0.5 * (horizontal_symmetry + vertical_symmetry), 0.0, 1.0)
    )

    yy, xx = np.indices(crop.shape, dtype=np.float32)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    best_fit = 0.0
    best_corner_empty = 0.0
    for scale in (0.78, 0.90, 1.00, 1.12, 1.25):
        half_width = max(1.0, (width - 1) * 0.5 * scale + 0.5)
        half_height = max(1.0, (height - 1) * 0.5 * scale + 0.5)
        l1_distance = (
            np.abs(xx - center_x) / half_width
            + np.abs(yy - center_y) / half_height
        )
        ideal = l1_distance <= 1.0
        union = int(np.count_nonzero(crop | ideal))
        if union:
            best_fit = max(
                best_fit, float(np.count_nonzero(crop & ideal) / union)
            )
        corner_zone = l1_distance >= 1.12
        if np.any(corner_zone):
            best_corner_empty = max(
                best_corner_empty, 1.0 - float(np.mean(crop[corner_zone]))
            )

    compactness = float(xs.size / max(1.0, width * height))
    extent_score = float(
        np.exp(-(((compactness - 0.56) / 0.32) ** 2))
    )
    base_score = (
        0.48 * best_fit
        + 0.20 * symmetry
        + 0.18 * best_corner_empty
        + 0.14 * extent_score
    )
    return float(
        np.clip(base_score * (0.18 + 0.82 * aspect_score), 0.0, 1.0)
    )


def stable_component_center(
    component_mask: np.ndarray,
    intensity: Optional[np.ndarray] = None,
) -> Point:
    """Находит геометрическое ядро компоненты без ухода к яркому хвосту.

    Обычный центр тяжести нестабилен для пересвеченного ромба: небольшое
    изменение экспозиции удлиняет один луч ореола и сдвигает маркер. Центр
    максимальной толщины маски (distance transform) остается на ядре.
    Геометрический центр и верхняя яркая площадка используются лишь как
    слабая субпиксельная поправка.
    """
    mask = np.asarray(component_mask, dtype=bool)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0.0, 0.0)

    geometric_center = np.array(
        [float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64
    )

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y1:y2, x1:x2].astype(np.uint8)
    distance_map = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    maximum_distance = float(np.max(distance_map))
    if maximum_distance > 0:
        # Берем всю вершину плато, а не один пиксель argmax.
        core_y, core_x = np.where(distance_map >= maximum_distance * 0.82)
        thickness_center = np.array(
            [
                float(np.mean(core_x)) + x1,
                float(np.mean(core_y)) + y1,
            ],
            dtype=np.float64,
        )
    else:
        thickness_center = geometric_center

    # Рамка компоненты особенно чувствительна к длинному тонкому лучу и в
    # итоговый центр не входит. Для симметричной компоненты thickness_center
    # и так совпадает с центром рамки; для компоненты с хвостом это различие
    # как раз защищает точку от ухода.
    center = 0.90 * thickness_center + 0.10 * geometric_center
    if intensity is not None:
        values = np.asarray(intensity, dtype=np.float32)
        if values.shape == mask.shape and xs.size >= 3:
            component_values = values[mask]
            high_level = float(np.percentile(component_values, 88.0))
            cap_y, cap_x = np.where(mask & (values >= high_level))
            if cap_x.size:
                cap_center = np.array(
                    [float(np.mean(cap_x)), float(np.mean(cap_y))],
                    dtype=np.float64,
                )
                center = 0.86 * center + 0.14 * cap_center

    return (float(center[0]), float(center[1]))


def _maximum_weight_unique_assignment(
    score_matrix: np.ndarray, minimum_score: float
) -> List[Tuple[int, int, float]]:
    """Находит глобально лучшее уникальное назначение строк кандидатам."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return []
    row_count, candidate_count = scores.shape

    # Отдельный фиктивный столбец для каждой области позволяет оставить Pn
    # неподтвержденной вместо назначения заведомо плохого кандидата.
    dummy_score = float(minimum_score) - 1e-6
    extended = np.full(
        (row_count, candidate_count + row_count), dummy_score, dtype=np.float64
    )
    extended[:, :candidate_count] = scores
    finite = extended[np.isfinite(extended)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    costs = maximum - np.where(np.isfinite(extended), extended, -1e6)

    # Венгерский алгоритм для прямоугольной матрицы (строк не больше столбцов).
    column_count = costs.shape[1]
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    p = np.zeros(column_count + 1, dtype=np.int32)
    way = np.zeros(column_count + 1, dtype=np.int32)
    for row in range(1, row_count + 1):
        p[0] = row
        column0 = 0
        minimum_values = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        while True:
            used[column0] = True
            current_row = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = (
                    costs[current_row - 1, column - 1]
                    - u[current_row]
                    - v[column]
                )
                if current < minimum_values[column]:
                    minimum_values[column] = current
                    way[column] = column0
                if minimum_values[column] < delta:
                    delta = minimum_values[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assigned_columns = np.full(row_count, -1, dtype=np.int32)
    for column in range(1, column_count + 1):
        if p[column] > 0:
            assigned_columns[p[column] - 1] = column - 1

    result = []
    for region_index, candidate_index in enumerate(assigned_columns):
        if 0 <= candidate_index < candidate_count:
            score = float(scores[region_index, candidate_index])
            if np.isfinite(score) and score >= minimum_score:
                result.append((region_index, int(candidate_index), score))
    return result


def assign_ir_candidates_to_regions(
    reference_regions: List[Rect], candidates: List[Dict]
) -> List[Tuple[int, Dict]]:
    """Назначает ИК-ромбы по сигнатуре Day→Night и свободному маршруту.

    Для каждой пары Pn-кандидат строится отрезок от центра исходной рамки.
    Если он входит в чужую базовую рамку, пара запрещается. Оставшиеся пары
    сравниваются одновременно, поэтому один кандидат не может достаться двум
    точкам. Расстояние и попадание в локальную область используются только
    как разрешение ничьей: настоящий дальний ИК-ромб не проигрывает более
    близкому краю белой поверхности.
    """
    if not reference_regions or not candidates:
        return []

    centers = []
    diagonals = []
    for x, y, width, height in reference_regions:
        centers.append(np.array([x + width / 2.0, y + height / 2.0]))
        diagonals.append(max(30.0, float(np.hypot(width, height))))

    score_matrix = np.full(
        (len(reference_regions), len(candidates)), -np.inf, dtype=np.float64
    )
    for candidate_index, candidate in enumerate(candidates):
        position = np.asarray(candidate["position"], dtype=np.float64)
        hints = set(candidate.get("region_hints", set()))
        route_diagnostics = {}
        for region_index in range(len(reference_regions)):
            x, y, width, height = reference_regions[region_index]
            distance = float(np.linalg.norm(position - centers[region_index]))
            blockers = route_blockers(
                tuple(centers[region_index]),
                tuple(position),
                reference_regions,
                region_index,
            )
            if blockers:
                route_diagnostics[str(region_index + 1)] = {
                    "distance_px": distance,
                    "blocked_by": [index + 1 for index in blockers],
                    "allowed": False,
                }
                continue

            length_scale = max(30.0, 1.35 * diagonals[region_index])
            length_score = float(
                np.exp(-((distance / length_scale) ** 2))
            )
            inside = float(
                x <= position[0] <= x + width
                and y <= position[1] <= y + height
            )
            if not hints:
                local_evidence = 0.45
            elif region_index in hints:
                local_evidence = 1.0
            else:
                local_evidence = 0.10
            quality = float(np.clip(candidate.get("quality", 0.0), 0.0, 1.0))
            signature = float(
                np.clip(candidate.get("signature_score", quality), 0.0, 1.0)
            )
            tracking_diamond = float(
                np.clip(
                    candidate.get(
                        "tracking_diamond_score",
                        candidate.get("diamond_score", 0.0),
                    ),
                    0.0,
                    1.0,
                )
            )
            # Все кандидаты к этому моменту уже дважды подтвердили физический
            # переход «черный Day → яркий ромб Night». Поэтому для нескольких
            # прошедших кандидатов возвращаем вес исходной области и длины
            # маршрута: валидный ромб внутри P2 должен победить столь же яркую
            # листву/деталь далеко за рамкой.
            line_weight = (
                0.72 * signature
                + 0.08 * tracking_diamond
                + 0.20 * length_score
            )
            score = (
                0.78 * line_weight
                + 0.12 * local_evidence
                + 0.06 * quality
                + 0.04 * inside
            )
            route_diagnostics[str(region_index + 1)] = {
                "distance_px": distance,
                "length_score": length_score,
                "signature_score": signature,
                "tracking_diamond_score": tracking_diamond,
                "line_weight": line_weight,
                "assignment_score": score,
                "blocked_by": [],
                "allowed": True,
            }
            score_matrix[region_index, candidate_index] = score
        candidate["assignment_routes"] = route_diagnostics

    assignments = _maximum_weight_unique_assignment(score_matrix, 0.40)
    matches = []
    for region_index, candidate_index, assignment_score in assignments:
        candidate = candidates[candidate_index]
        candidate["region_index"] = region_index
        candidate["assignment_score"] = float(assignment_score)
        selected_route = candidate.get("assignment_routes", {}).get(
            str(region_index + 1), {}
        )
        candidate["assignment_distance_px"] = float(
            selected_route.get("distance_px", 0.0)
        )
        candidate["assignment_line_weight"] = float(
            selected_route.get("line_weight", 0.0)
        )
        candidate["assignment_origin"] = [
            float(centers[region_index][0]),
            float(centers[region_index][1]),
        ]
        matches.append((region_index, candidate))
    matches.sort(key=lambda item: item[0])
    return matches


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
    diamond_score: float = 0.0


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
    # Нулевая координата задается только первым реальным LOCK после запуска.
    # Центр стартовой области и сохраненная ИК-привязка ею не считаются.
    initial_position: Optional[Point] = None
    displacement: Point = (0.0, 0.0)
    displacement_magnitude: float = 0.0


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
        for name, value in BASE_DISPLAY_SETTINGS.items():
            setattr(self, name, value)

        # Параметры одноразового поиска по отклику ИК-подсветки Hikvision.
        self.ir_settle_seconds = 2.0
        self.ir_flash_delta = 25
        # Максимальная исходная яркость ядра в Day: отражатель должен быть
        # практически черным, а не просто темнее окружающего фона.
        self.ir_day_black_threshold = 85
        self.ir_search_scale = 4.0
        self.ir_sample_count = 5
        self.hikvision_channel = 1
        # В строгом режиме каждый Pn анализируется независимо только рядом со
        # своей исходной областью. Это исключает перестановку отражателей и
        # захват сильного кандидата из соседней области.
        self.ir_strict_regions = True
        # Если локальная область оказалась задана неточно, пропущенная точка
        # дополнительно ищется по всему кадру, но только по признакам
        # «темный Day → яркий Night» и ромбовидному ИК-ореолу.
        self.ir_global_fallback = True
        self.ir_diamond_min_score = 0.45
        self.ir_lock_enabled = True
        # Локальный радиус поиска от последнего подтвержденного положения.
        self.ir_lock_radius = 12.0
        # Допустимый полный ход отражателя от стартовой Day→Night-точки.
        self.ir_max_travel = 100.0
        # Ручные исходные области хранятся отдельно: ошибочная автоматическая
        # привязка больше не становится отправной точкой следующего ИК-поиска.
        self.ir_reference_regions: List[Rect] = []
        self.ir_verification_active = False
        self.ir_model_version = 10
        self.ir_confirmed_centers: Dict[str, List[float]] = {}
        self.ir_confirmed_models: Dict[str, Dict] = {}

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
            "show_displacements": self.show_displacements,
            "close_shape": self.close_shape,
            "ir_settle_seconds": self.ir_settle_seconds,
            "ir_flash_delta": self.ir_flash_delta,
            "ir_day_black_threshold": self.ir_day_black_threshold,
            "ir_search_scale": self.ir_search_scale,
            "ir_sample_count": self.ir_sample_count,
            "hikvision_channel": self.hikvision_channel,
            "ir_strict_regions": self.ir_strict_regions,
            "ir_global_fallback": self.ir_global_fallback,
            "ir_diamond_min_score": self.ir_diamond_min_score,
            "ir_lock_enabled": self.ir_lock_enabled,
            "ir_lock_radius": self.ir_lock_radius,
            "ir_max_travel": self.ir_max_travel,
            "ir_reference_regions": self.ir_reference_regions,
            "ir_verification_active": self.ir_verification_active,
            "ir_model_version": self.ir_model_version,
            "ir_confirmed_centers": self.ir_confirmed_centers,
            "ir_confirmed_models": self.ir_confirmed_models,
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
            "show_displacements",
            "close_shape",
            "ir_settle_seconds",
            "ir_flash_delta",
            "ir_day_black_threshold",
            "ir_search_scale",
            "ir_sample_count",
            "hikvision_channel",
            "ir_strict_regions",
            "ir_global_fallback",
            "ir_diamond_min_score",
            "ir_lock_enabled",
            "ir_lock_radius",
            "ir_max_travel",
            "ir_verification_active",
        ):
            if name in data:
                setattr(self, name, data[name])

        self.ir_lock_radius = float(
            np.clip(float(self.ir_lock_radius), 3.0, 100.0)
        )
        self.ir_day_black_threshold = int(
            np.clip(int(self.ir_day_black_threshold), 10, 160)
        )
        self.ir_max_travel = float(
            np.clip(
                max(float(self.ir_max_travel), self.ir_lock_radius),
                10.0,
                500.0,
            )
        )

        self.roi_region = self._region(data.get("roi_region"))
        regions = []
        for region in data.get("peak_regions", []):
            parsed = self._region(region)
            if parsed is not None:
                regions.append(parsed)
        self.peak_regions = regions

        reference_regions = []
        for region in data.get("ir_reference_regions", []):
            parsed = self._region(region)
            if parsed is not None:
                reference_regions.append(parsed)
        self.ir_reference_regions = (
            reference_regions
            if len(reference_regions) == len(self.peak_regions)
            else list(self.peak_regions)
        )
        if reference_regions and len(reference_regions) == len(self.peak_regions):
            # Версии до 5.14 переносили peak_regions после ИК-поиска. При
            # загрузке возвращаем сохраненные ручные области, которые и
            # определяют непересекающиеся территории P1…Pn.
            self.peak_regions = list(reference_regions)
            xs = [region[0] for region in self.peak_regions]
            ys = [region[1] for region in self.peak_regions]
            x2s = [region[0] + region[2] for region in self.peak_regions]
            y2s = [region[1] + region[3] for region in self.peak_regions]
            self.roi_region = (
                min(xs),
                min(ys),
                max(x2s) - min(xs),
                max(y2s) - min(ys),
            )

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

        try:
            loaded_ir_model_version = int(data.get("ir_model_version", 0))
        except (TypeError, ValueError):
            loaded_ir_model_version = 0

        self.ir_confirmed_centers = {}
        raw_centers = data.get("ir_confirmed_centers", {})
        if loaded_ir_model_version >= self.ir_model_version and isinstance(raw_centers, dict):
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

        self.ir_confirmed_models = {}
        raw_models = data.get("ir_confirmed_models", {})
        if loaded_ir_model_version >= self.ir_model_version and isinstance(raw_models, dict):
            for raw_id, raw_model in raw_models.items():
                try:
                    region_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if region_id < 1 or not isinstance(raw_model, dict):
                    continue
                cleaned_model = {}
                for name in (
                    "area",
                    "diamond_score",
                    "quality",
                    "response",
                    "night_peak",
                    "halo_radius",
                    "tracking_area",
                    "tracking_diamond_score",
                    "tracking_radius",
                ):
                    try:
                        value = float(raw_model.get(name, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value >= 0:
                        cleaned_model[name] = value
                for name in (
                    "tracking_center_offset_x",
                    "tracking_center_offset_y",
                ):
                    try:
                        value = float(raw_model.get(name, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        cleaned_model[name] = value
                bbox = self._region(raw_model.get("bbox"))
                if bbox is not None:
                    cleaned_model["bbox"] = list(bbox)
                if cleaned_model:
                    self.ir_confirmed_models[str(region_id)] = cleaned_model

        # Старые якоря были получены до проверки маршрутов к кандидатам.
        # Они намеренно не переносятся: иначе ошибочно назначенный соседний
        # отражатель мог бы снова получить LOCK после загрузки настроек.
        if loaded_ir_model_version < self.ir_model_version and raw_centers:
            self.ir_confirmed_centers.clear()
            self.ir_confirmed_models.clear()
            self.ir_verification_active = True


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
                "User-Agent": "ReflectorTracker/5.16",
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
        use_background_difference: bool = True,
    ) -> np.ndarray:
        x, y, w, h = rect
        crop = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if (
            use_background_difference
            and self.background is not None
            and self.background.shape == frame.shape[:2]
        ):
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
        anchor_model: Optional[Dict] = None,
    ) -> Optional[PeakDetection]:
        rect = clip_rect(region, frame.shape)
        if rect is None:
            return None
        x0, y0, width, height = rect
        # Подтвержденный ИК-отражатель виден по собственной яркости. Разность
        # с фоновым кадром здесь запрещена: движение тени или автомобиля иначе
        # превращается в искусственный яркий сигнал и уводит измеряемый центр.
        signal = self._signal(
            frame,
            rect,
            detection_settings,
            use_background_difference=anchor is None,
        )
        signal_float = signal.astype(np.float32)
        background_level = self._border_level(signal)
        peak_value = float(np.max(signal_float))
        local_contrast = peak_value - background_level
        detection_peak = peak_value
        detection_contrast = local_contrast
        decision_signal = signal_float
        anchor_probe_local = None
        if anchor is not None:
            anchor_probe_local = np.array(
                [anchor[0] - x0, anchor[1] - y0], dtype=np.float32
            )
            probe_radius = max(3.0, float(anchor_radius or max(width, height)))
            probe_x1 = max(0, int(np.floor(anchor_probe_local[0] - probe_radius)))
            probe_y1 = max(0, int(np.floor(anchor_probe_local[1] - probe_radius)))
            probe_x2 = min(width, int(np.ceil(anchor_probe_local[0] + probe_radius + 1)))
            probe_y2 = min(height, int(np.ceil(anchor_probe_local[1] + probe_radius + 1)))
            if probe_x2 > probe_x1 and probe_y2 > probe_y1:
                decision_signal = signal_float[probe_y1:probe_y2, probe_x1:probe_x2]
                detection_peak = float(np.max(decision_signal))
                detection_contrast = detection_peak - background_level

        brightness = float(
            self._setting(detection_settings, "brightness_threshold")
        )
        contrast = max(
            1.0, float(self._setting(detection_settings, "contrast_threshold"))
        )
        adaptive = bool(
            self._setting(detection_settings, "adaptive_threshold")
        )

        absolute_ok = detection_peak >= brightness
        adaptive_ok = (
            adaptive
            and detection_peak >= max(20.0, brightness * 0.65)
            and detection_contrast >= contrast * 1.5
        )
        # После подтверждения Day→Night разрешаем более слабую яркость внутри
        # малого радиуса якоря. ИК-отражатель может мерцать из-за экспозиции,
        # однако далёкий белый объект по-прежнему недоступен этому треку.
        anchor_ok = (
            anchor is not None
            and detection_peak >= max(70.0, brightness * 0.45)
            and detection_contrast >= max(6.0, contrast * 0.35)
        )
        required_contrast = max(6.0, contrast * 0.35) if anchor_ok else contrast
        if (
            not (absolute_ok or adaptive_ok or anchor_ok)
            or detection_contrast < required_contrast
        ):
            return None

        if anchor_ok:
            # Для подтвержденного ИК-якоря сначала выделяем яркое ядро.
            # Этот режим имеет приоритет даже при включенном адаптивном пороге.
            # Низкий порог объединял ромб отражателя с белым откосом/крышей:
            # центр общей компоненты уходил за допуск и правильная точка
            # получала LOST, хотя визуально продолжала ярко светиться.
            relative_core_level = background_level + 0.62 * detection_contrast
            peak_core_level = detection_peak - max(
                12.0, 0.32 * detection_contrast
            )
            threshold = max(
                background_level + 0.35 * contrast,
                relative_core_level,
                peak_core_level,
            )
        elif adaptive:
            percentile = float(
                np.percentile(
                    decision_signal,
                    np.clip(
                        self._setting(
                            detection_settings, "brightness_percentile"
                        ),
                        80.0,
                        99.95,
                    ),
                )
            )
            shape_level = background_level + 0.38 * detection_contrast
            allowed_floor = brightness if absolute_ok else brightness * 0.65
            threshold = max(
                allowed_floor,
                background_level + 0.5 * contrast,
                min(percentile, shape_level),
            )
        else:
            threshold = brightness
        threshold = min(threshold, detection_peak - 1.0)

        binary = np.where(signal_float >= threshold, 255, 0).astype(np.uint8)

        # При сопровождении подтвержденного отражателя анализ ограничивается
        # окрестностью его неизменного Day→Night-якоря. Большая область поиска
        # нужна на случай потери, но не должна присоединять к блику удаленные
        # пересвеченные поверхности.
        anchor_support_radius = 0.0
        if anchor_probe_local is not None:
            model_halo_radius = 0.0
            if anchor_model:
                try:
                    model_halo_radius = max(
                        0.0, float(anchor_model.get("halo_radius", 0.0))
                    )
                except (TypeError, ValueError):
                    model_halo_radius = 0.0
            anchor_support_radius = max(
                18.0,
                float(anchor_radius or 0.0) * 2.5,
                float(anchor_radius or 0.0)
                + max(8.0, model_halo_radius * 2.2),
            )
            support_y, support_x = np.indices(signal.shape, dtype=np.float32)
            support_mask = (
                (support_x - anchor_probe_local[0]) ** 2
                + (support_y - anchor_probe_local[1]) ** 2
                <= anchor_support_radius**2
            )
            binary[~support_mask] = 0
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
        # Площадь маски Day→Night response не равна площади яркого ядра на
        # обычном Night-кадре. Для строгого сравнения используем только
        # tracking_area, измеренную тем же детектором; старое поле area служит
        # исключительно описанием ИК-вспышки.
        expected_anchor_area = 0.0
        expected_anchor_diamond = 0.0
        response_anchor_diamond = 0.0
        if anchor_model:
            try:
                expected_anchor_area = max(
                    0.0, float(anchor_model.get("tracking_area", 0.0))
                )
            except (TypeError, ValueError):
                expected_anchor_area = 0.0
            try:
                expected_anchor_diamond = float(
                    np.clip(
                        anchor_model.get("tracking_diamond_score", 0.0),
                        0.0,
                        1.0,
                    )
                )
            except (TypeError, ValueError):
                expected_anchor_diamond = 0.0
            try:
                response_anchor_diamond = float(
                    np.clip(anchor_model.get("diamond_score", 0.0), 0.0, 1.0)
                )
            except (TypeError, ValueError):
                response_anchor_diamond = 0.0
        if expected_anchor_area > 0:
            # Общий max_area относится к обычному яркому ядру. У
            # подтвержденной призмы ромбовидный ИК-ореол может быть в несколько
            # раз больше (как у P2 на предоставленном Night-кадре).
            max_area = max(max_area, int(np.ceil(expected_anchor_area * 4.0)))
        elif anchor_probe_local is not None:
            # Первый Night-кадр после ИК-поиска калибрует площадь сопровождения.
            # До этого не ограничиваем его величиной response-маски.
            max_area = max(
                max_area,
                int(np.ceil(np.pi * max(18.0, anchor_support_radius) ** 2)),
            )
        candidates = []
        positive = np.maximum(signal_float - background_level, 0.0)
        power = float(
            np.clip(
                self._setting(detection_settings, "center_power"), 1.0, 5.0
            )
        )
        if anchor is not None:
            # Слишком большая степень смещает центр по единственному
            # насыщенному пикселю и вызывает дрожание точки.
            power = min(power, 2.0)
        powered = np.power(positive, power)
        predicted_local = None
        if predicted is not None:
            predicted_local = np.array([predicted[0] - x0, predicted[1] - y0])
        anchor_local = anchor_probe_local
        allowed_anchor_radius = max(3.0, float(anchor_radius or 0.0))
        distance_scale = max(10.0, 0.35 * np.hypot(width, height))

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            component_mask = labels == label
            diamond_similarity = 1.0
            current_diamond = 0.0
            if anchor_local is not None:
                current_diamond = diamond_halo_score(component_mask)
                minimum_tracking_diamond = 0.12
                if expected_anchor_diamond > 0:
                    minimum_tracking_diamond = max(
                        minimum_tracking_diamond,
                        min(0.38, expected_anchor_diamond * 0.38),
                    )
                if current_diamond < minimum_tracking_diamond:
                    continue
                reference_diamond = (
                    expected_anchor_diamond
                    if expected_anchor_diamond > 0
                    else response_anchor_diamond
                )
                if reference_diamond <= 0:
                    reference_diamond = current_diamond
                diamond_similarity = float(
                    np.exp(
                        -abs(current_diamond - reference_diamond) / 0.45
                    )
                )
            energy = float(np.sum(powered[component_mask]))
            if anchor_local is not None:
                # Предварительный допуск также считаем по устойчивому ядру,
                # а не по центроиду всего ореола. Иначе светлый хвост способен
                # ошибочно вынести реально переместившийся отражатель за
                # разрешенный радиус ещё до точного расчета центра.
                center = np.asarray(
                    stable_component_center(component_mask, signal_float),
                    dtype=np.float64,
                )
            else:
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
            area_similarity = 1.0
            if expected_anchor_area > 0:
                area_ratio = area / expected_anchor_area
                # Размер ореола меняется с экспозицией, поэтому допускаем
                # широкий диапазон, но крупная крыша не должна быть похожа на
                # небольшую подтвержденную призму.
                if area_ratio < 0.12 or area_ratio > 8.0:
                    continue
                area_similarity = float(
                    np.exp(-abs(np.log(max(area_ratio, 1e-6))) / 1.15)
                )
            # Без ИК-якоря сохраняется прежнее поведение. При наличии якоря
            # посторонний большой белый объект больше не может победить только
            # за счет площади и энергии.
            predicted_weight = 0.30 + 0.70 * proximity
            if anchor_local is not None:
                # Логарифм энергии не даёт крупному пересвеченному фрагменту
                # победить небольшой отражатель только за счёт площади.
                score = (
                    np.log1p(max(energy, 0.0))
                    * (0.10 + 0.90 * anchor_proximity)
                    * predicted_weight
                    * (0.22 + 0.78 * area_similarity)
                    * (0.25 + 0.75 * diamond_similarity)
                )
            else:
                score = energy * predicted_weight
            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            combined_proximity = float(np.sqrt(proximity * anchor_proximity))
            candidates.append(
                (score, label, bbox, combined_proximity, current_diamond)
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_label, best_bbox, best_proximity, best_diamond = candidates[0]

        # Дополнительно присоединяем соседние компоненты. Это важно, когда
        # пересвеченная призма содержит несколько разорванных ярких участков.
        selected_labels = {best_label}
        selected_boxes = [best_bbox]
        join_distance = max(1.0, merge_radius * 1.5)
        changed = True
        while changed:
            changed = False
            for _, label, bbox, _, _ in candidates:
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
            background_level + 0.18 * detection_contrast,
            min(threshold, brightness * 0.65),
        )
        final_mask = (neighborhood > 0) & (signal_float >= halo_threshold)
        final_area = int(np.count_nonzero(final_mask))
        if final_area < min_area:
            return None

        measurement_mask = final_mask
        measurement_threshold = float(halo_threshold)
        tracking_area = float(final_area)
        if anchor_local is not None:
            # Форма всего ореола нужна только для проверки кандидата. Центр
            # подтвержденного отражателя измеряем по наиболее яркому
            # компактному ядру внутри ЛУЧШЕЙ компоненты. Край ореола, тень и
            # светлая поверхность рядом больше не участвуют в координате.
            best_component = labels == best_label
            best_component_values = signal_float[best_component]
            if not best_component_values.size:
                return None
            component_peak = float(np.max(best_component_values))
            component_contrast = max(1.0, component_peak - background_level)
            core_levels = (
                max(
                    threshold,
                    background_level + 0.84 * component_contrast,
                    float(np.percentile(best_component_values, 84.0)),
                ),
                max(
                    threshold,
                    background_level + 0.72 * component_contrast,
                    float(np.percentile(best_component_values, 70.0)),
                ),
                float(threshold),
            )
            bright_core = best_component
            for core_level in core_levels:
                proposed_core = best_component & (signal_float >= core_level)
                if np.count_nonzero(proposed_core) >= 3:
                    bright_core = proposed_core
                    measurement_threshold = float(core_level)
                    break

            core_count, core_labels, core_stats, _ = (
                cv2.connectedComponentsWithStats(
                    bright_core.astype(np.uint8), connectivity=8
                )
            )
            reference_local = (
                predicted_local if predicted_local is not None else anchor_local
            )
            selected_core = None
            selected_core_score = -np.inf
            core_distance_scale = max(2.0, allowed_anchor_radius * 0.40)
            for core_label in range(1, core_count):
                core_area = int(
                    core_stats[core_label, cv2.CC_STAT_AREA]
                )
                if core_area < 1:
                    continue
                core_component = core_labels == core_label
                core_x, core_y = stable_component_center(
                    core_component, signal_float
                )
                core_distance = float(
                    np.linalg.norm(
                        np.asarray([core_x, core_y]) - reference_local
                    )
                )
                proximity_score = float(
                    np.exp(-((core_distance / core_distance_scale) ** 2))
                )
                core_peak = float(np.max(signal_float[core_component]))
                core_peak_score = float(
                    np.clip(
                        (core_peak - background_level) / component_contrast,
                        0.0,
                        1.0,
                    )
                )
                core_area_score = float(
                    np.clip(core_area / 8.0, 0.0, 1.0)
                )
                core_score = (
                    0.55 * proximity_score
                    + 0.35 * core_peak_score
                    + 0.10 * core_area_score
                )
                if core_score > selected_core_score:
                    selected_core_score = core_score
                    selected_core = core_component
            if selected_core is not None:
                measurement_mask = selected_core
            else:
                measurement_mask = best_component

            tracking_area = float(np.count_nonzero(best_component))
            stable_x, stable_y = stable_component_center(
                measurement_mask, signal_float
            )
            center_x = stable_x + x0
            center_y = stable_y + y0
        else:
            weights = powered * final_mask.astype(np.float32)
            total_weight = float(np.sum(weights))
            if total_weight <= 0:
                return None
            yy, xx = np.indices(signal.shape, dtype=np.float32)
            center_x = float(np.sum(xx * weights) / total_weight) + x0
            center_y = float(np.sum(yy * weights) / total_weight) + y0

        measurement_ys, measurement_xs = np.where(measurement_mask)
        if not measurement_xs.size:
            return None
        center_area = int(measurement_xs.size)
        mask_u8 = measurement_mask.astype(np.uint8) * 255
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

        bbox_global = (
            int(measurement_xs.min() + x0),
            int(measurement_ys.min() + y0),
            int(measurement_xs.max() - measurement_xs.min() + 1),
            int(measurement_ys.max() - measurement_ys.min() + 1),
        )
        radius = float(np.sqrt(center_area / np.pi))

        contrast_score = float(
            np.clip(detection_contrast / (contrast * 3.0), 0.0, 1.0)
        )
        brightness_score = float(
            np.clip(detection_peak / max(brightness, 1.0), 0.0, 1.0)
        )
        area_score = float(
            np.clip(tracking_area / max(min_area * 2.0, 1.0), 0.0, 1.0)
        )
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
            area=tracking_area,
            circularity=circularity,
            bbox=bbox_global,
            threshold=measurement_threshold,
            max_intensity=detection_peak,
            diamond_score=best_diamond,
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
            saved_anchor = self.calibration.ir_confirmed_centers.get(
                str(index + 1)
            )
            if saved_anchor and len(saved_anchor) == 2:
                center = (float(saved_anchor[0]), float(saved_anchor[1]))
            saved_model = self.calibration.ir_confirmed_models.get(
                str(index + 1), {}
            )
            try:
                initial_area = max(
                    0.0, float(saved_model.get("tracking_area", 0.0))
                )
            except (TypeError, ValueError):
                initial_area = 0.0
            has_verified_anchor = bool(saved_anchor and len(saved_anchor) == 2)
            initial_confidence = float(
                np.clip(
                    saved_model.get(
                        "quality", saved_model.get("diamond_score", 0.0)
                    ),
                    0.0,
                    1.0,
                )
            )
            initial_radius = max(
                3.0,
                float(
                    saved_model.get(
                        "tracking_radius", saved_model.get("halo_radius", 5.0)
                    )
                ),
            )
            initial_search_scale = 1.0
            if has_verified_anchor:
                margin = max(4.0, initial_radius + 3.0)
                required_scale = max(
                    1.0,
                    2.0 * (abs(center[0] - (x + w / 2.0)) + margin) / w,
                    2.0 * (abs(center[1] - (y + h / 2.0)) + margin) / h,
                )
                initial_search_scale = min(
                    max(1.0, float(self.calibration.roi_max_scale)),
                    required_scale,
                )
            track = ReflectorPoint(
                id=index + 1,
                base_region=(x, y, w, h),
                position=center,
                radius=initial_radius,
                confidence=initial_confidence,
                last_seen=time.time() if has_verified_anchor else 0.0,
                is_active=has_verified_anchor,
                has_measurement=has_verified_anchor,
                measured_this_frame=False,
                detection_area=initial_area,
                search_scale=initial_search_scale,
            )
            self.tracks.append(track)
            history = deque(maxlen=self.history_length)
            if has_verified_anchor:
                history.append(center)
            self.position_history[index + 1] = history
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

    def _has_verified_ir_anchor(self, track: ReflectorPoint) -> bool:
        saved = self.calibration.ir_confirmed_centers.get(str(track.id))
        return bool(
            self.calibration.ir_lock_enabled
            and saved
            and len(saved) == 2
        )

    def _ir_dynamic_search_radius(self, track: ReflectorPoint) -> float:
        """Локальный радиус растет после потери, но не превышает полный ход."""
        base_radius = float(
            np.clip(self.calibration.ir_lock_radius, 3.0, 100.0)
        )
        maximum_travel = float(
            np.clip(
                max(self.calibration.ir_max_travel, base_radius),
                10.0,
                500.0,
            )
        )
        if track.missed_frames <= 0:
            return base_radius
        expansion_per_frame = max(4.0, base_radius * 0.35)
        return min(
            maximum_travel,
            base_radius + expansion_per_frame * track.missed_frames,
        )

    def _exclusive_search_regions(
        self, frame_shape: Tuple[int, ...]
    ) -> List[Rect]:
        reference_regions = [track.base_region for track in self.tracks]
        proposed_regions = []
        for track in self.tracks:
            x, y, width, height = track.base_region
            base_center = (x + width / 2.0, y + height / 2.0)
            center = base_center
            scale = float(track.search_scale)
            if self._has_verified_ir_anchor(track) and track.has_measurement:
                # После подтверждения область следует только за последним
                # надежным положением своего отражателя. При потере она
                # остается там же и расширяется, а не возвращается к центру
                # исходной рамки и не прыгает вслед за случайным бликом.
                center = track.position
                dynamic_radius = self._ir_dynamic_search_radius(track)
                margin = max(4.0, track.radius + 3.0)
                scale = max(
                    scale,
                    2.0 * (dynamic_radius + margin) / max(3.0, width),
                    2.0 * (dynamic_radius + margin) / max(3.0, height),
                )
            proposed_regions.append(
                self._centered_region(
                    center,
                    track.base_region,
                    scale,
                    frame_shape,
                )
            )
        return make_regions_exclusive(
            proposed_regions, reference_regions, frame_shape
        )

    def _update_track(
        self,
        track: ReflectorPoint,
        frame: np.ndarray,
        now: float,
        search_region: Optional[Rect] = None,
    ):
        base_x, base_y, base_w, base_h = track.base_region
        base_center = (base_x + base_w / 2.0, base_y + base_h / 2.0)
        predicted = (
            track.position[0] + track.velocity[0],
            track.position[1] + track.velocity[1],
        )
        if not track.has_measurement:
            predicted = base_center

        # До Day→Night-подтверждения область остается у исходной рамки. После
        # подтверждения она движется только за последним надежным ядром.
        if search_region is None:
            search_center = base_center
            search_scale = float(track.search_scale)
            if self._has_verified_ir_anchor(track) and track.has_measurement:
                search_center = track.position
                dynamic_radius = self._ir_dynamic_search_radius(track)
                margin = max(4.0, track.radius + 3.0)
                search_scale = max(
                    search_scale,
                    2.0 * (dynamic_radius + margin) / max(3.0, base_w),
                    2.0 * (dynamic_radius + margin) / max(3.0, base_h),
                )
            search_region = self._centered_region(
                search_center,
                track.base_region,
                search_scale,
                frame.shape,
            )
        detection_settings = self.calibration.region_settings.get(str(track.id))
        ir_anchor = None
        ir_anchor_radius = None
        ir_max_travel = None
        ir_anchor_model = None
        ir_tracking_offset = np.zeros(2, dtype=np.float64)
        detector_anchor = None
        if self.calibration.ir_lock_enabled:
            saved_anchor = self.calibration.ir_confirmed_centers.get(str(track.id))
            if saved_anchor and len(saved_anchor) == 2:
                ir_anchor = (float(saved_anchor[0]), float(saved_anchor[1]))
                ir_anchor_model = self.calibration.ir_confirmed_models.get(
                    str(track.id)
                )
                ir_anchor_radius = self._ir_dynamic_search_radius(track)
                ir_max_travel = float(
                    np.clip(
                        max(
                            self.calibration.ir_max_travel,
                            self.calibration.ir_lock_radius,
                        ),
                        10.0,
                        500.0,
                    )
                )
                if ir_anchor_model:
                    try:
                        ir_tracking_offset = np.array(
                            [
                                float(
                                    ir_anchor_model.get(
                                        "tracking_center_offset_x", 0.0
                                    )
                                ),
                                float(
                                    ir_anchor_model.get(
                                        "tracking_center_offset_y", 0.0
                                    )
                                ),
                            ],
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError):
                        ir_tracking_offset = np.zeros(2, dtype=np.float64)
                maximum_offset = max(
                    2.0,
                    min(
                        8.0,
                        float(self.calibration.ir_lock_radius) * 0.75,
                    ),
                )
                offset_length = float(np.linalg.norm(ir_tracking_offset))
                if not np.all(np.isfinite(ir_tracking_offset)) or (
                    offset_length > maximum_offset
                ):
                    ir_tracking_offset = np.zeros(2, dtype=np.float64)
                # Локальный детектор следует за последним подтвержденным
                # положением. Исходный Day→Night-якорь остается только
                # абсолютным нулём и границей максимального полного хода.
                detector_anchor_array = (
                    np.asarray(track.position, dtype=np.float64)
                    + ir_tracking_offset
                )
                detector_anchor = (
                    float(detector_anchor_array[0]),
                    float(detector_anchor_array[1]),
                )
        needs_ir_confirmation = (
            self.calibration.ir_verification_active
            and self.calibration.ir_lock_enabled
            and ir_anchor is None
        )
        if needs_ir_confirmation:
            # После завершенного ИК-поиска запрещено подменять пропущенную
            # метку любым ярким объектом обычного кадра. Такая точка остается
            # NO IR до повторного подтверждения Day→Night.
            detection = None
        else:
            detector_predicted = predicted
            if ir_anchor is not None:
                detector_predicted = (
                    float(track.position[0] + ir_tracking_offset[0]),
                    float(track.position[1] + ir_tracking_offset[1]),
                )
            detection = self.detector.detect_in_region(
                frame,
                search_region,
                detector_predicted,
                detection_settings=detection_settings,
                anchor=detector_anchor,
                anchor_radius=ir_anchor_radius,
                anchor_model=ir_anchor_model,
            )
            if detection is not None and ir_anchor is not None:
                # Детектор Night и Day→Night могут иметь постоянное смещение
                # центра на 1–3 px из-за разной бинаризации одного ореола.
                # Оно измеряется при калибровке и вычитается до фильтрации;
                # поэтому первый LOCK не «прыгает», а реальные перемещения
                # отражателя сохраняются без мертвой зоны.
                detection.position = (
                    float(detection.position[0] - ir_tracking_offset[0]),
                    float(detection.position[1] - ir_tracking_offset[1]),
                )

        # Day→Night-центр остается неподвижным нулем измерений, но не центром
        # каждого локального поиска. Отражатель может пройти 50–80 px; только
        # выход за отдельный предел полного хода считается неверным захватом.
        if detection is not None and ir_anchor is not None:
            anchor_error = float(
                np.linalg.norm(
                    np.asarray(detection.position) - np.asarray(ir_anchor)
                )
            )
            if anchor_error > float(ir_max_travel):
                detection = None

        # В обычном режиме действует строгий лимит скачка. После потери поиск
        # разрешается во всей текущей (но неподвижной) области, при этом новый
        # блик должен быть похож по площади на ранее отслеживаемый.
        if detection is not None and track.has_measurement:
            reference = (
                track.position
                if ir_anchor is not None or track.missed_frames > 0
                else predicted
            )
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
                    if ir_anchor is not None:
                        minimum_area_ratio, maximum_area_ratio = 0.20, 5.00
                    else:
                        minimum_area_ratio, maximum_area_ratio = 0.40, 2.50
                    if (
                        area_ratio < minimum_area_ratio
                        or area_ratio > maximum_area_ratio
                    ):
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
                if ir_anchor is not None:
                    # Для подтвержденной неподвижной ИК-метки скорость не
                    # должна сама сдвигать координату. Фильтруем только новое
                    # измерение яркого ядра; реальные медленные перемещения
                    # сохраняются, а единичный уход по тени не накапливается.
                    filtered = (
                        (1.0 - alpha) * old_position
                        + alpha * measurement
                    )
                    instantaneous = measurement - old_position
                    velocity = (
                        0.85 * old_velocity + 0.15 * instantaneous
                    )
                else:
                    prediction = old_position + old_velocity
                    filtered = (
                        (1.0 - alpha) * prediction
                        + alpha * measurement
                    )
                    instantaneous = filtered - old_position
                    velocity = (
                        0.70 * old_velocity + 0.30 * instantaneous
                    )

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
            if track.initial_position is None:
                # Первый подтвержденный кадр после запуска — абсолютный ноль
                # для всех последующих перемещений этой точки.
                track.initial_position = track.position
            displacement_x = track.position[0] - track.initial_position[0]
            displacement_y = track.position[1] - track.initial_position[1]
            track.displacement = (displacement_x, displacement_y)
            track.displacement_magnitude = float(
                np.hypot(displacement_x, displacement_y)
            )
            if ir_anchor_model is not None:
                # Модель обычного Night-сопровождения обучается отдельно от
                # Day→Night response-маски. Плавное обновление учитывает
                # автоэкспозицию, но не позволяет площади измениться скачком.
                previous_tracking_area = float(
                    ir_anchor_model.get("tracking_area", 0.0) or 0.0
                )
                previous_tracking_diamond = float(
                    ir_anchor_model.get("tracking_diamond_score", 0.0) or 0.0
                )
                previous_tracking_radius = float(
                    ir_anchor_model.get("tracking_radius", 0.0) or 0.0
                )
                model_alpha = 0.12
                ir_anchor_model["tracking_area"] = (
                    detection.area
                    if previous_tracking_area <= 0
                    else (1.0 - model_alpha) * previous_tracking_area
                    + model_alpha * detection.area
                )
                ir_anchor_model["tracking_diamond_score"] = (
                    detection.diamond_score
                    if previous_tracking_diamond <= 0
                    else (1.0 - model_alpha) * previous_tracking_diamond
                    + model_alpha * detection.diamond_score
                )
                ir_anchor_model["tracking_radius"] = (
                    detection.radius
                    if previous_tracking_radius <= 0
                    else (1.0 - model_alpha) * previous_tracking_radius
                    + model_alpha * detection.radius
                )
            shrinking_scale = max(
                1.0,
                track.search_scale - max(0.12, (track.search_scale - 1.0) * 0.35),
            )

            if ir_anchor is not None:
                # Центр следующего окна уже будет на найденной точке, поэтому
                # нет необходимости сохранять огромную рамку от старта до
                # текущего положения.
                track.search_scale = min(
                    max(1.0, float(self.calibration.roi_max_scale)),
                    shrinking_scale,
                )
            else:
                # В обычном режиме окно остается привязано к базовой области.
                margin = max(4.0, track.radius + 3.0)
                required_scale = max(
                    1.0,
                    2.0
                    * (abs(track.position[0] - base_center[0]) + margin)
                    / base_w,
                    2.0
                    * (abs(track.position[1] - base_center[1]) + margin)
                    / base_h,
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
            if ir_anchor is not None:
                # Для резкого переноса на десятки пикселей рамка должна
                # раскрыться за несколько кадров, а не за десятки секунд.
                scale_step = max(scale_step, 0.12)
            track.search_scale = min(
                max(1.0, float(self.calibration.roi_max_scale)),
                track.search_scale + scale_step,
            )

        track.search_region = search_region

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.update_settings()
        now = time.time()
        search_regions = self._exclusive_search_regions(frame.shape)
        for index, track in enumerate(self.tracks):
            search_region = (
                search_regions[index]
                if index < len(search_regions)
                else None
            )
            self._update_track(track, frame, now, search_region)

        # Масштаб мог измениться после LOCK/HOLD. Готовим непересекающиеся
        # рамки следующего кадра и показываем именно их.
        next_search_regions = self._exclusive_search_regions(frame.shape)
        for index, track in enumerate(self.tracks):
            if index < len(next_search_regions):
                track.search_region = next_search_regions[index]

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
            needs_ir_confirmation = (
                calibration.ir_verification_active
                and calibration.ir_lock_enabled
                and not uses_ir_anchor
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
                elif needs_ir_confirmation:
                    frame_label += " NO IR"
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
                if (
                    calibration.show_displacements
                    and track.initial_position is not None
                ):
                    initial_x = int(round(track.initial_position[0]))
                    initial_y = int(round(track.initial_position[1]))
                    displacement_color = (255, 0, 255)
                    cv2.drawMarker(
                        display,
                        (initial_x, initial_y),
                        displacement_color,
                        cv2.MARKER_CROSS,
                        11,
                        1,
                        cv2.LINE_AA,
                    )
                    if initial_x != px or initial_y != py:
                        cv2.line(
                            display,
                            (initial_x, initial_y),
                            (px, py),
                            displacement_color,
                            1,
                            cv2.LINE_AA,
                        )
                    label_x = max(
                        5, min(max(5, display.shape[1] - 245), px + 9)
                    )
                    label_y = max(15, min(display.shape[0] - 8, py + 16))
                    outlined_text(
                        (
                            f"dX:{track.displacement[0]:+.2f} "
                            f"dY:{track.displacement[1]:+.2f} "
                            f"dR:{track.displacement_magnitude:.2f} px"
                        ),
                        (label_x, label_y),
                        0.40,
                        displacement_color,
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
        self.root.title("Система устойчивого отслеживания отражателей v5.24")
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
        self._ir_day_verify_samples = []
        self._ir_night_verify_samples = []
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
        self._view_crop_rect = (0, 0, 1, 1)
        self._view_image_origin = (0.0, 0.0)
        self._view_frame_shape = (1, 1)

        # Модальный выбор областей выполняется прямо на основном Canvas.
        self._region_selection_active = False
        self._region_selection_frame: Optional[np.ndarray] = None
        self._region_selection_expected = 0
        self._region_selection_rects: List[Rect] = []
        self._region_selection_start: Optional[Point] = None
        self._region_selection_current: Optional[Point] = None
        self._region_selection_previous_regions: List[Rect] = []
        self._region_selection_escape_bind_id = None
        self._layer_drawer_open = False

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
            cursor = (
                "crosshair"
                if self._region_selection_active
                else ("fleur" if self.view_zoom > 1.0 else "")
            )
            self.main_canvas.config(cursor=cursor)

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
            self.main_canvas.config(
                cursor="crosshair" if self._region_selection_active else ""
            )

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

    def _canvas_to_frame(
        self, canvas_x: float, canvas_y: float, clamp_to_image: bool = False
    ) -> Optional[Point]:
        """Переводит координату Canvas в исходный кадр с учетом zoom/pan."""
        origin_x, origin_y = self._view_image_origin
        render_w, render_h = self._view_render_size
        if render_w <= 0 or render_h <= 0:
            return None
        local_x = float(canvas_x) - origin_x
        local_y = float(canvas_y) - origin_y
        if clamp_to_image:
            local_x = self._clamp(local_x, 0.0, float(render_w))
            local_y = self._clamp(local_y, 0.0, float(render_h))
        elif not (0.0 <= local_x <= render_w and 0.0 <= local_y <= render_h):
            return None
        crop_x, crop_y, crop_w, crop_h = self._view_crop_rect
        frame_h, frame_w = self._view_frame_shape
        frame_x = crop_x + local_x / max(1.0, render_w) * crop_w
        frame_y = crop_y + local_y / max(1.0, render_h) * crop_h
        return (
            self._clamp(frame_x, 0.0, max(0.0, frame_w - 1.0)),
            self._clamp(frame_y, 0.0, max(0.0, frame_h - 1.0)),
        )

    def _frame_to_canvas(self, frame_x: float, frame_y: float) -> Point:
        crop_x, crop_y, crop_w, crop_h = self._view_crop_rect
        origin_x, origin_y = self._view_image_origin
        render_w, render_h = self._view_render_size
        return (
            origin_x + (float(frame_x) - crop_x) / max(1.0, crop_w) * render_w,
            origin_y + (float(frame_y) - crop_y) / max(1.0, crop_h) * render_h,
        )

    def _canvas_button_press(self, event):
        if not self._region_selection_active:
            self._start_pan(event)
            return
        point = self._canvas_to_frame(event.x, event.y)
        if point is None:
            return "break"
        self._region_selection_start = point
        self._region_selection_current = point
        return "break"

    def _canvas_drag(self, event):
        if not self._region_selection_active:
            self._pan_view(event)
            return
        if self._region_selection_start is not None:
            point = self._canvas_to_frame(
                event.x, event.y, clamp_to_image=True
            )
            if point is not None:
                self._region_selection_current = point
        return "break"

    def _canvas_button_release(self, event):
        if not self._region_selection_active:
            self._end_pan(event)
            return
        if self._region_selection_start is None:
            return "break"
        end = self._canvas_to_frame(event.x, event.y, clamp_to_image=True)
        start = self._region_selection_start
        self._region_selection_start = None
        self._region_selection_current = None
        if end is None:
            return "break"
        x1 = int(round(min(start[0], end[0])))
        y1 = int(round(min(start[1], end[1])))
        x2 = int(round(max(start[0], end[0])))
        y2 = int(round(max(start[1], end[1])))
        region = clip_rect(
            (x1, y1, x2 - x1, y2 - y1),
            self._region_selection_frame.shape,
        )
        if region is None:
            self.status_label.config(
                text="Область слишком мала — протяните рамку не менее 3×3 px"
            )
            return "break"
        self._region_selection_rects.append(region)
        if len(self._region_selection_rects) >= self._region_selection_expected:
            self._finish_peak_region_selection()
        else:
            self._update_region_selection_status()
        return "break"

    def _canvas_double_click(self, event):
        if self._region_selection_active:
            return "break"
        self.reset_view_zoom()
        return "break"

    def _create_layer_drawer(self):
        """Создает выдвижной флажок слоев поверх правого края камеры."""
        self.layer_drawer = ttk.Frame(
            self.display_frame, relief=tk.RAISED, borderwidth=2, padding=10
        )
        header = ttk.Frame(self.layer_drawer)
        header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(header, text="Слои отображения", font=("Arial", 10, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Button(
            header, text="▶", width=3, command=self._toggle_layer_drawer
        ).pack(side=tk.RIGHT)
        layer_options = (
            ("Точки и подписи", self.show_points_var),
            ("Круги отражателей", self.show_circles_var),
            ("Рамки областей", self.show_frames_var),
            ("Линии между точками", self.show_lines_var),
            ("Расстояния по линиям", self.show_distances_var),
            ("Изменение расстояний dL", self.show_distance_changes_var),
            ("Смещение от старта dX/dY/dR", self.show_displacements_var),
            ("Замкнуть контур", self.close_shape_var),
        )
        for label, variable in layer_options:
            ttk.Checkbutton(
                self.layer_drawer,
                text=label,
                variable=variable,
                command=self._on_display_layer_change,
            ).pack(anchor=tk.W, fill=tk.X, pady=2)
        ToolTip(
            self.layer_drawer,
            "Слои независимы. dL — изменение длины между кадрами; "
            "dX/dY/dR — смещение от первого LOCK после запуска.",
        )
        self.layer_drawer_tab = ttk.Button(
            self.display_frame,
            text="◀\nСлои",
            command=self._toggle_layer_drawer,
        )
        self.layer_drawer_tab.place(
            relx=1.0, x=-39, y=72, width=39, height=92
        )

    def _toggle_layer_drawer(self):
        self._layer_drawer_open = not self._layer_drawer_open
        drawer_width = 275
        if self._layer_drawer_open:
            self.layer_drawer.place(
                relx=1.0, x=-drawer_width, y=72, width=drawer_width
            )
            self.layer_drawer_tab.config(text="▶")
            self.layer_drawer_tab.place_configure(
                relx=1.0, x=-(drawer_width + 39), y=72, width=39, height=92
            )
            self.layer_drawer.lift()
            self.layer_drawer_tab.lift()
        else:
            self.layer_drawer.place_forget()
            self.layer_drawer_tab.config(text="◀\nСлои")
            self.layer_drawer_tab.place_configure(
                relx=1.0, x=-39, y=72, width=39, height=92
            )
            self.layer_drawer_tab.lift()

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
        black_row = ttk.Frame(hikvision)
        black_row.pack(fill=tk.X, pady=2)
        self.ir_day_black_var = tk.IntVar(
            value=self.calibration.ir_day_black_threshold
        )
        ttk.Label(
            black_row, text="Макс. яркость черного ядра в Day:"
        ).pack(side=tk.LEFT)
        day_black_spin = ttk.Spinbox(
            black_row,
            from_=10,
            to=160,
            increment=5,
            textvariable=self.ir_day_black_var,
            width=6,
        )
        day_black_spin.pack(side=tk.RIGHT)
        day_black_spin.bind("<Return>", self._on_ir_lock_change)
        day_black_spin.bind("<FocusOut>", self._on_ir_lock_change)
        strict_row = ttk.Frame(hikvision)
        strict_row.pack(fill=tk.X, pady=2)
        self.ir_strict_regions_var = tk.BooleanVar(
            value=self.calibration.ir_strict_regions
        )
        ttk.Checkbutton(
            strict_row,
            text="Локальный поиск отдельно для P1…Pn",
            variable=self.ir_strict_regions_var,
            command=self._on_ir_lock_change,
        ).pack(side=tk.LEFT)
        diamond_row = ttk.Frame(hikvision)
        diamond_row.pack(fill=tk.X, pady=2)
        self.ir_global_fallback_var = tk.BooleanVar(
            value=self.calibration.ir_global_fallback
        )
        self.ir_diamond_min_score_var = tk.DoubleVar(
            value=self.calibration.ir_diamond_min_score
        )
        ttk.Checkbutton(
            diamond_row,
            text="Резервный поиск ромба по кадру",
            variable=self.ir_global_fallback_var,
            command=self._on_ir_lock_change,
        ).pack(side=tk.LEFT)
        diamond_spin = ttk.Spinbox(
            diamond_row,
            from_=0.10,
            to=0.90,
            increment=0.05,
            textvariable=self.ir_diamond_min_score_var,
            width=5,
        )
        diamond_spin.pack(side=tk.RIGHT)
        diamond_spin.bind("<Return>", self._on_ir_lock_change)
        diamond_spin.bind("<FocusOut>", self._on_ir_lock_change)
        ttk.Label(diamond_row, text="Мин. D:").pack(side=tk.RIGHT, padx=(3, 2))
        lock_row = ttk.Frame(hikvision)
        lock_row.pack(fill=tk.X, pady=2)
        self.ir_lock_enabled_var = tk.BooleanVar(
            value=self.calibration.ir_lock_enabled
        )
        self.ir_lock_radius_var = tk.DoubleVar(
            value=self.calibration.ir_lock_radius
        )
        self.ir_max_travel_var = tk.DoubleVar(
            value=self.calibration.ir_max_travel
        )
        ttk.Checkbutton(
            lock_row,
            text="Удерживать ИК-метку",
            variable=self.ir_lock_enabled_var,
            command=self._on_ir_lock_change,
        ).pack(side=tk.LEFT)
        ir_lock_radius_spin = ttk.Spinbox(
            lock_row,
            from_=3,
            to=100,
            increment=1,
            textvariable=self.ir_lock_radius_var,
            width=6,
        )
        ir_lock_radius_spin.pack(side=tk.RIGHT)
        ir_lock_radius_spin.bind("<Return>", self._on_ir_lock_change)
        ir_lock_radius_spin.bind("<FocusOut>", self._on_ir_lock_change)
        ttk.Label(lock_row, text="Локально, px:").pack(side=tk.RIGHT, padx=(3, 2))
        travel_row = ttk.Frame(hikvision)
        travel_row.pack(fill=tk.X, pady=2)
        ir_max_travel_spin = ttk.Spinbox(
            travel_row,
            from_=10,
            to=500,
            increment=10,
            textvariable=self.ir_max_travel_var,
            width=6,
        )
        ir_max_travel_spin.pack(side=tk.RIGHT)
        ir_max_travel_spin.bind("<Return>", self._on_ir_lock_change)
        ir_max_travel_spin.bind("<FocusOut>", self._on_ir_lock_change)
        ttk.Label(
            travel_row, text="Макс. ход от старта, px:"
        ).pack(side=tk.RIGHT, padx=(3, 2))
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
            text="🔦 Найти отражатели: Day ↔ Night ×2",
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
            "Процедура выполняет два полных цикла Day→Night и получает не менее "
            "пяти кадров каждого режима в каждом цикле. Кандидат принимается "
            "только если он оба раза темный в Day, яркий и ромбовидный в Night. "
            "Статичные детали исключаются. После поиска камера остается в Night.",
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

        # Переменные слоев создаются здесь, а сами переключатели размещаются
        # в выдвижной панели поверх правой стороны видеокадра.
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
        self.show_displacements_var = tk.BooleanVar(
            value=self.calibration.show_displacements
        )
        self.close_shape_var = tk.BooleanVar(value=self.calibration.close_shape)

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

        self.display_frame = ttk.Frame(main)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.main_canvas = tk.Canvas(
            self.display_frame, bg="black", highlightthickness=0
        )
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<MouseWheel>", self._zoom_wheel)
        self.main_canvas.bind("<Button-4>", self._zoom_wheel)
        self.main_canvas.bind("<Button-5>", self._zoom_wheel)
        self.main_canvas.bind("<ButtonPress-1>", self._canvas_button_press)
        self.main_canvas.bind("<B1-Motion>", self._canvas_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self._canvas_button_release)
        self.main_canvas.bind("<ButtonPress-2>", self._start_pan)
        self.main_canvas.bind("<B2-Motion>", self._pan_view)
        self.main_canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.main_canvas.bind("<Button-3>", self._undo_peak_region_selection)
        self.main_canvas.bind("<Double-Button-1>", self._canvas_double_click)
        self._create_layer_drawer()

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
   Разметка идет прямо на основном кадре с текущим приближением. ЛКМ рисует
   рамку, колесо меняет масштаб, средняя кнопка перемещает кадр, ПКМ отменяет
   последнюю рамку, Esc отменяет всю разметку без потери прежних областей.
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
Флажок «Слои» находится справа поверх кадра и открывает выдвижную панель.
Точки, круги, рамки, линии, расстояния, изменение расстояний и смещения от
старта включаются независимо. Точки соединяются по порядку P1—P2—P3...
«Замкнуть контур» добавляет линию от последней активной точки к P1, если точек
не меньше трех. dL — изменение длины ребра относительно предыдущего кадра.
dX, dY и dR считаются от первого подтвержденного LOCK после каждого запуска.

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
3. Нажмите «Найти отражатели: Day ↔ Night ×2». Камера выполнит два полных
   цикла Day→Night и возьмет не менее пяти кадров каждого режима в каждом
   цикле. Для каждого Pn независимо проверяются: темное ядро в обоих Day,
   яркое ромбовидное ядро в обоих Night и физический прирост исходной яркости
   при каждом включении ИК. Статичные поверхности исключаются до назначения
   номеров P1…Pn.
4. Зеленые метки — подтвержденные ИК-отражатели. D — сходство светового
   ореола с ромбом. Оранжевые рамки — локальные кандидаты, фиолетовые —
   кандидаты резервного поиска по всему кадру.
   S — итоговая достоверность локальной проверки Day/Night. От центра каждой
   исходной рамки строятся маршруты ко всем кандидатам. Маршрут исключается,
   если входит в чужую базовую рамку. Остальные сравниваются по длине L и весу
   W, учитывающему длину и качество ромба. Выбранный маршрут показан зеленой
   линией. После назначения центр уточняется по яркому ядру Night-кадра, а не
   по смещенной разности Day→Night.
   Красная рамка X:NIGHT_CORE означает отклонение при уточнении яркого ядра;
   X:SIG_CORE — не удалось выделить компактное Night-ядро;
   X:SIG_NIGHT — ядро недостаточно яркое в Night;
   X:SIG_DAY — ядро уже слишком яркое в Day;
   X:SIG_NOT_BLACK — в Day не найдено совпадающее черное ядро;
   X:SIG_DAY_UNSTABLE — черное Day-ядро найдено в разных местах двух циклов;
   X:SIG_GAIN — недостаточный ИК-прирост тех же пикселей ядра;
   X:SIG_STATIC — яркость не повторяет оба переключения Day/Night;
   X:SIG_LOCAL — слабый локальный контраст без сильной вспышки ядра.
5. Рекомендуется включить «Локальный поиск отдельно для P1…Pn». Тогда каждая
   область получает собственный порог. Флажок «Резервный поиск ромба по кадру»
   позволяет найти отражатель за пределами ранее ошибочно перенесенной области.
   Рекомендуемое «Мин. D» — 0,40–0,55. Если отражатель не найден, уменьшите
   «Мин. вспышка» (например, с 25 до 15) или «Мин. D» на 0,05.
   «Макс. яркость черного ядра в Day» по умолчанию равна 85 из 255. Уменьшайте
   ее, если нужно требовать более глубокий черный; увеличивайте только если
   настоящий отражатель получает X:SIG_NOT_BLACK из-за сжатия видеопотока.
   Черное Day-ядро ищется отдельно в окрестности Night-ромба с автоматическим
   допуском примерно 10–24 px на изменение фокуса/масштаба IR-cut.
После процедуры камера остается в Night для обычного сопровождения. Постоянное
переключение механического IR-cut фильтра во время каждого кадра не выполняется.
Флажок «Удерживать ИК-метку» не позволяет треку перескочить на более яркий
объект за пределами допуска. Рекомендуемый допуск — 8–12 px. Если текущий
ромб не прошёл проверку координаты, площади или формы, используется HOLD без
переноса маркера.
В режиме Night сначала выделяется яркое ядро внутри ИК-якоря. Поэтому ореол
отражателя не объединяется с белой крышей или освещённым откосом. Площадь
Night-ядра калибруется отдельно от площади Day→Night-вспышки.
Если после Day→Night конкретная точка не подтверждена, она получает состояние
NO IR и не может быть заменена крышей, бордюром или другим ярким объектом.
Каждый поиск автоматически сохраняет в папку ir_scan_records рядом со скриптом
кадры Day, Night, изображение результата и JSON с координатами Pn.

LOCK — пик найден на текущем кадре.
HOLD — блик временно не распознан, но последняя точка удерживается.
LOST — превышено заданное число кадров удержания.

При HOLD/LOST поиск остается привязанным к исходной области и автоматически
расширяется максимум примерно до двойного размера. Если отражатель найден за
исходной границей, область сохраняет достаточный размер для его сопровождения.
Если исходные области P1…Pn не пересекаются, их расширенные поисковые области
также разделяются и не имеют общей зоны. Один блик не может одновременно стать
кандидатом двух соседних треков.
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
            "Система обнаружения отражателей v5.24\n\n"
            "• один постоянный трек на одну заданную область\n"
            "• объединение сложного блика в единую точку\n"
            "• яркостно-взвешенный субпиксельный центр\n"
            "• адаптивное расширение области при потере\n"
            "• масштабирование и перемещение изображения\n"
            "• разметка областей в основном окне с сохранением масштаба\n"
            "• запись окна в MP4 и журнал состояний CSV\n"
            "• встроенные и постоянные пользовательские пресеты\n"
            "• закреплённый поиск с расширением до двойной области\n"
            "• непересекающиеся поисковые территории P1…Pn\n"
            "• независимые слои геометрии и покадровые изменения расстояний\n"
            "• нулевая координата первого LOCK и смещения dX/dY/dR\n"
            "• выдвижная панель слоев справа от видеокадра\n"
            "• индивидуальные настройки обнаружения для P1…Pn\n"
            "• ИК-поиск Hikvision по признаку «темный Day → яркий Night»\n"
            "• отдельный локальный порог и список кандидатов для каждого Pn\n"
            "• назначение по свободному маршруту, длине L и весу W\n"
            "• проверка ромбовидного ИК-ореола и резервный поиск по кадру\n"
            "• проверка Day/Night по маске ядра без разбавления светлым фоном\n"
            "• двойная проверка Day→Night→Day→Night против статичных деталей\n"
            "• отдельный порог компактного черного ядра в Day\n"
            "• независимое сопоставление смещенных Day- и Night-ядер\n"
            "• глобально-оптимальное назначение кандидатов областям P1…Pn\n"
            "• уточнение центра по яркому ядру Night после Day→Night\n"
            "• независимая двойная проверка каждого Pn в Day и Night\n"
            "• медиана не менее пяти кадров в каждом режиме\n"
            "• запрет ложного LOCK для точки без подтверждения Day→Night\n"
            "• передача подтвержденного центра в трекер как первого измерения\n"
            "• отдельная модель яркого ядра для обычного Night-сопровождения\n"
            "• отсечение слияния отражателя с белым откосом или крышей\n"
            "• автоматическое сохранение Day/Night и результата ИК-поиска\n"
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

        if len(self.calibration.ir_reference_regions) != len(
            self.calibration.peak_regions
        ):
            self.calibration.ir_reference_regions = list(
                self.calibration.peak_regions
            )

        self.hikvision_control = control
        self._ir_scan_running = True
        self._ir_scan_token += 1
        token = self._ir_scan_token
        self._ir_day_samples = []
        self._ir_night_samples = []
        self._ir_day_verify_samples = []
        self._ir_night_verify_samples = []
        self.ir_scan_btn.config(state=tk.DISABLED, text="Переключение в Day…")
        self.hik_status_label.config(
            text="Этап 1/8: включение Day (цикл 1)", foreground="#b05a00"
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
                text=f"Этап 2/8: Day-кадры цикла 1 ({delay / 1000:.1f} с)",
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
        required = max(5, int(self.calibration.ir_sample_count))
        if len(target) >= required:
            done(token)
            return
        self.root.after(140, lambda: self._collect_ir_samples(target, token, done))

    def _switch_ir_scan_to_night(self, token: int):
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        self.ir_scan_btn.config(text="Переключение в Night…")
        self.hik_status_label.config(
            text="Этап 3/8: Night и ИК-подсветка (цикл 1)",
            foreground="#b05a00",
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
                    self._switch_ir_scan_to_day_verify,
                ),
            )

        self._background_call(
            lambda: self.hikvision_control.set_ircut("night"),
            night_ready,
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=True),
        )

    def _switch_ir_scan_to_day_verify(self, token: int):
        """Второй Day обязателен: статичный объект должен остаться ярким/темным."""
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        self.ir_scan_btn.config(text="Повторное переключение в Day…")
        self.hik_status_label.config(
            text="Этап 4/8: повторный Day (проверка исчезновения вспышки)",
            foreground="#b05a00",
        )

        def day_ready(_):
            if token != self._ir_scan_token:
                return
            delay = int(round(self.calibration.ir_settle_seconds * 1000))
            self.ir_scan_btn.config(text="Повторная проверка Day…")

            def collect_day_verify():
                self.hik_status_label.config(
                    text="Этап 5/8: запись Day-кадров цикла 2",
                    foreground="#b05a00",
                )
                self._collect_ir_samples(
                    self._ir_day_verify_samples,
                    token,
                    self._switch_ir_scan_to_night_verify,
                )

            self.root.after(
                delay,
                collect_day_verify,
            )

        self._background_call(
            lambda: self.hikvision_control.set_ircut("day"),
            day_ready,
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=True),
        )

    def _switch_ir_scan_to_night_verify(self, token: int):
        """Второй Night подтверждает повторяемую ИК-вспышку в том же месте."""
        if token != self._ir_scan_token or not self._ir_scan_running:
            return
        self.ir_scan_btn.config(text="Повторное переключение в Night…")
        self.hik_status_label.config(
            text="Этап 6/8: повторный Night и ИК-подсветка",
            foreground="#b05a00",
        )

        def night_ready(_):
            if token != self._ir_scan_token:
                return
            delay = int(round(self.calibration.ir_settle_seconds * 1000))
            self.ir_scan_btn.config(text="Повторная проверка Night…")

            def collect_night_verify():
                self.hik_status_label.config(
                    text="Этап 7/8: запись Night-кадров цикла 2",
                    foreground="#b05a00",
                )
                self._collect_ir_samples(
                    self._ir_night_verify_samples,
                    token,
                    self._analyze_ir_scan_async,
                )

            self.root.after(
                delay,
                collect_night_verify,
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
            text="Этап 8/8: сравнение двух циклов и привязка P1…Pn",
            foreground="#b05a00",
        )
        day_samples = list(self._ir_day_samples)
        night_samples = list(self._ir_night_samples)
        day_verify_samples = list(self._ir_day_verify_samples)
        night_verify_samples = list(self._ir_night_verify_samples)
        self._background_call(
            lambda: self._analyze_ir_flash_frames(
                day_samples,
                night_samples,
                day_verify_samples,
                night_verify_samples,
            ),
            lambda result: self._finish_ir_flash_scan(token, result),
            lambda exc: self._abort_ir_flash_scan(exc, restore_mode=False),
        )

    @staticmethod
    def _align_ir_day_frame(day: np.ndarray, night: np.ndarray):
        """Совмещает Day с Night по устойчивым границам объектов.

        Переключение IR-cut меняет не только яркость, но иногда и масштаб/
        фокус. Поэтому сначала пробуем ограниченное аффинное совмещение, а
        затем безопасный сдвиг. Некорректное преобразование отклоняется.
        """
        try:
            height, width = day.shape
            registration_scale = min(1.0, 900.0 / max(height, width))
            if registration_scale < 1.0:
                registration_size = (
                    max(96, int(round(width * registration_scale))),
                    max(96, int(round(height * registration_scale))),
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

            def edge_image(image):
                low, high = np.percentile(image, (3.0, 97.0))
                span = max(1.0, float(high - low))
                normalized = np.clip((image - low) / span, 0.0, 1.0).astype(
                    np.float32
                )
                normalized = cv2.GaussianBlur(normalized, (0, 0), 1.1)
                grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
                magnitude = cv2.magnitude(grad_x, grad_y)
                scale = max(1e-6, float(np.percentile(magnitude, 98.0)))
                return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)

            day_edges = edge_image(day_registration)
            night_edges = edge_image(night_registration)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                120,
                1e-6,
            )
            best = None
            for motion in (cv2.MOTION_AFFINE, cv2.MOTION_TRANSLATION):
                warp = np.eye(2, 3, dtype=np.float32)
                try:
                    correlation, warp = cv2.findTransformECC(
                        night_edges,
                        day_edges,
                        warp,
                        motion,
                        criteria,
                    )
                except Exception:
                    continue
                shift_x = float(warp[0, 2] / registration_scale)
                shift_y = float(warp[1, 2] / registration_scale)
                linear = warp[:, :2].astype(np.float64)
                scale_x = float(np.linalg.norm(linear[:, 0]))
                scale_y = float(np.linalg.norm(linear[:, 1]))
                determinant = float(np.linalg.det(linear))
                rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
                shear = float(
                    abs(np.dot(linear[:, 0], linear[:, 1]))
                    / max(1e-6, scale_x * scale_y)
                )
                valid = (
                    np.isfinite(correlation)
                    and correlation >= 0.16
                    and abs(shift_x) <= 60.0
                    and abs(shift_y) <= 60.0
                    and 0.94 <= scale_x <= 1.06
                    and 0.94 <= scale_y <= 1.06
                    and determinant > 0.85
                    and abs(rotation) <= 3.0
                    and shear <= 0.06
                )
                if not valid:
                    continue
                full_warp = warp.copy()
                full_warp[0, 2] = shift_x
                full_warp[1, 2] = shift_y
                if best is None or correlation > best[0]:
                    best = (float(correlation), full_warp, shift_x, shift_y)

            if best is None:
                return day, (0.0, 0.0, 0.0)
            correlation, full_warp, shift_x, shift_y = best
            aligned = cv2.warpAffine(
                day,
                full_warp,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, (shift_x, shift_y, correlation)
        except Exception:
            return day, (0.0, 0.0, 0.0)

    @staticmethod
    def _diamond_halo_score(component_mask: np.ndarray) -> float:
        return diamond_halo_score(component_mask)

    def _extract_ir_candidates(
        self,
        response: np.ndarray,
        day_normalized: np.ndarray,
        night_normalized: np.ndarray,
        night: np.ndarray,
        search_rect: Rect,
        detection_settings: Optional[Dict] = None,
        region_hint: Optional[int] = None,
        source: str = "local",
    ):
        """Выделяет подтвержденные Day→Night ромбовидные кандидаты."""
        clipped = clip_rect(search_rect, response.shape)
        if clipped is None:
            return [], float(self.calibration.ir_flash_delta)
        sx, sy, sw, sh = clipped
        local_response = response[sy : sy + sh, sx : sx + sw]
        positive_values = local_response[local_response > 0]
        if not positive_values.size:
            return [], float(self.calibration.ir_flash_delta)

        percentile_level = float(np.percentile(positive_values, 98.8))
        maximum_response = float(np.max(local_response))
        adaptive_level = min(percentile_level, maximum_response * 0.24)
        local_threshold = max(
            float(self.calibration.ir_flash_delta), adaptive_level
        )
        binary = np.where(local_response >= local_threshold, 255, 0).astype(
            np.uint8
        )
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        minimum_area = max(
            2,
            int(
                float(
                    (detection_settings or {}).get(
                        "min_area", self.calibration.min_area
                    )
                )
                // 2
            ),
        )
        maximum_area = max(
            500,
            int(
                float(
                    (detection_settings or {}).get(
                        "max_area", self.calibration.max_area
                    )
                )
                * 8
            ),
        )
        brightness = float(
            (detection_settings or {}).get(
                "brightness_threshold", self.calibration.brightness_threshold
            )
        )
        minimum_diamond = float(
            np.clip(self.calibration.ir_diamond_min_score, 0.05, 0.95)
        )
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area or area > maximum_area:
                continue
            component = labels == label
            diamond_score = self._diamond_halo_score(component)
            if diamond_score < minimum_diamond:
                continue

            local_ys, local_xs = np.where(component)
            global_xs = local_xs + sx
            global_ys = local_ys + sy
            response_values = local_response[local_ys, local_xs]
            peak_response = float(np.max(response_values))
            night_values = night[global_ys, global_xs]
            night_peak = float(np.max(night_values))
            day_level = float(np.mean(day_normalized[global_ys, global_xs]))
            night_level = float(np.mean(night_normalized[global_ys, global_xs]))
            relative_flash = night_level - day_level
            if night_peak < max(65.0, brightness * 0.35):
                continue
            # Абсолютную темноту Day окончательно проверяем по двум циклам
            # исходной яркости. Глобальная нормализация здесь не имеет права
            # заранее удалить реальный темный отражатель возле светлой крыши.
            if relative_flash <= 0.07:
                continue

            bbox = (
                int(stats[label, cv2.CC_STAT_LEFT]) + sx,
                int(stats[label, cv2.CC_STAT_TOP]) + sy,
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            local_center = stable_component_center(
                component, local_response
            )
            center = (
                float(local_center[0] + sx),
                float(local_center[1] + sy),
            )

            gain_score = float(np.clip(relative_flash / 0.55, 0.0, 1.0))
            response_score = float(
                np.clip(peak_response / max(20.0, local_threshold * 3.0), 0.0, 1.0)
            )
            darkness_score = float(np.clip((0.86 - day_level) / 0.60, 0.0, 1.0))
            night_score = float(np.clip((night_level - 0.30) / 0.65, 0.0, 1.0))
            quality = float(
                np.clip(
                    0.38 * diamond_score
                    + 0.24 * gain_score
                    + 0.18 * response_score
                    + 0.10 * darkness_score
                    + 0.10 * night_score,
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                {
                    "position": center,
                    "bbox": bbox,
                    "area": area,
                    "response": peak_response,
                    "strength": float(peak_response * np.sqrt(area)),
                    "quality": quality,
                    "day_level": day_level,
                    "night_level": night_level,
                    "night_peak": night_peak,
                    "diamond_score": diamond_score,
                    "halo_radius": float(np.sqrt(area / np.pi)),
                    "local_threshold": float(local_threshold),
                    "region_hints": ({region_hint} if region_hint is not None else set()),
                    "source": source,
                }
            )
        candidates.sort(
            key=lambda item: (item["quality"], item["strength"]), reverse=True
        )
        return candidates, float(local_threshold)

    def _refine_ir_candidate_on_night(
        self,
        night_u8: np.ndarray,
        candidate: Dict,
        region_index: int,
    ) -> bool:
        """Уточняет центр только по локальному ядру исходного response-ромба."""
        x, y, width, height = [int(value) for value in candidate["bbox"]]
        response_center = (
            float(candidate["position"][0]),
            float(candidate["position"][1]),
        )
        candidate["response_position"] = response_center
        candidate["refined_night_core"] = False
        object_size = max(3.0, float(max(width, height)))
        support_radius = max(12.0, min(32.0, object_size * 0.90))
        probe_radius = max(4.0, min(12.0, object_size * 0.40))
        margin = int(np.ceil(support_radius + 3.0))
        crop_rect = clip_rect(
            (
                int(round(response_center[0])) - margin,
                int(round(response_center[1])) - margin,
                2 * margin + 1,
                2 * margin + 1,
            ),
            night_u8.shape,
        )
        if crop_rect is None:
            return False
        crop_x, crop_y, crop_width, crop_height = crop_rect
        gray = cv2.GaussianBlur(
            night_u8[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width],
            (3, 3),
            0.5,
        ).astype(np.float32)
        local_center = np.array(
            [response_center[0] - crop_x, response_center[1] - crop_y],
            dtype=np.float32,
        )
        yy, xx = np.indices(gray.shape, dtype=np.float32)
        distance = np.sqrt(
            (xx - local_center[0]) ** 2 + (yy - local_center[1]) ** 2
        )
        probe_values = gray[distance <= probe_radius]
        ring_values = gray[
            (distance >= support_radius * 0.72)
            & (distance <= support_radius)
        ]
        if probe_values.size < 4 or ring_values.size < 12:
            return False
        peak = float(np.percentile(probe_values, 98.0))
        background = float(np.median(ring_values))
        local_contrast = peak - background
        if local_contrast < 8.0:
            return False
        threshold = max(
            background + 0.65 * local_contrast,
            peak - max(10.0, 0.25 * local_contrast),
        )
        binary = (
            (gray >= threshold) & (distance <= support_radius)
        ).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        maximum_correction = max(6.0, min(12.0, object_size * 0.42))
        minimum_diamond = max(
            0.18,
            min(0.38, float(self.calibration.ir_diamond_min_score) * 0.45),
        )
        components = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            correction = float(
                np.linalg.norm(centroids[label] - local_center)
            )
            if correction > maximum_correction:
                continue
            if area < 3 or area > max(80, int(width * height * 2.5)):
                continue
            component = labels == label
            diamond = diamond_halo_score(component)
            if diamond < minimum_diamond:
                continue
            distance_score = float(
                np.exp(
                    -(
                        correction
                        / max(3.0, maximum_correction * 0.55)
                    )
                    ** 2
                )
            )
            score = distance_score * (0.30 + 0.70 * diamond)
            components.append((score, label, correction, diamond, area))
        if not components:
            return False

        _, best_label, correction, diamond, area = max(components)
        component = labels == best_label
        local_position = np.asarray(
            stable_component_center(component, gray), dtype=np.float64
        )
        position = local_position + np.array([crop_x, crop_y])
        bbox = (
            int(stats[best_label, cv2.CC_STAT_LEFT]) + crop_x,
            int(stats[best_label, cv2.CC_STAT_TOP]) + crop_y,
            int(stats[best_label, cv2.CC_STAT_WIDTH]),
            int(stats[best_label, cv2.CC_STAT_HEIGHT]),
        )
        candidate["position"] = (float(position[0]), float(position[1]))
        candidate["center_correction"] = float(correction)
        candidate["tracking_area"] = float(area)
        candidate["tracking_diamond_score"] = float(diamond)
        candidate["tracking_radius"] = float(np.sqrt(area / np.pi))
        candidate["tracking_bbox"] = [int(value) for value in bbox]
        # Сохраняем именно пиксели найденного яркого ядра. Проверка Day/Night
        # ниже должна сравнивать один и тот же отражатель, а не усреднять
        # большой круг, в который возле P2 часто попадает светлый откос.
        core_y, core_x = np.where(component)
        candidate["_tracking_core_pixels"] = np.column_stack(
            (core_x + crop_x, core_y + crop_y)
        ).astype(np.int32)
        candidate["refined_night_core"] = True
        candidate["night_core_threshold"] = float(threshold)
        return True

    @staticmethod
    def _validate_ir_local_signature(
        day_normalized: np.ndarray,
        night_normalized: np.ndarray,
        candidate: Dict,
        day_cycles_raw: Optional[List[np.ndarray]] = None,
        night_cycles_raw: Optional[List[np.ndarray]] = None,
        day_black_threshold: int = 85,
    ) -> bool:
        """Проверяет ИК-переход по маске реально найденного Night-ядра.

        Круг вокруг центра ненадежен возле светлой крыши/откоса: фон занимает
        большинство круга и разбавляет вспышку маленького ромба. Поэтому
        основной признак — прирост тех же пикселей ядра Day→Night. Узкое
        кольцо вокруг ядра используется только как дополнительное
        подтверждение локального контраста, но не может само погубить сильную
        вспышку отражателя.
        """
        candidate["signature_failure"] = "SIG_CORE"
        if not bool(candidate.get("refined_night_core", False)):
            return False

        raw_pixels = candidate.get("_tracking_core_pixels")
        if raw_pixels is None:
            return False
        pixels = np.asarray(raw_pixels, dtype=np.int32)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] < 3:
            return False

        height, width = day_normalized.shape
        valid_pixels = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        pixels = pixels[valid_pixels]
        if pixels.shape[0] < 3:
            return False

        bbox = candidate.get("tracking_bbox", candidate["bbox"])
        object_size = max(3.0, float(max(bbox[2], bbox[3])))
        ring_radius = int(round(max(4.0, min(12.0, object_size * 0.70))))
        x1 = max(0, int(np.min(pixels[:, 0])) - ring_radius - 1)
        y1 = max(0, int(np.min(pixels[:, 1])) - ring_radius - 1)
        x2 = min(width, int(np.max(pixels[:, 0])) + ring_radius + 2)
        y2 = min(height, int(np.max(pixels[:, 1])) + ring_radius + 2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return False

        core = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        core[pixels[:, 1] - y1, pixels[:, 0] - x1] = 1
        # Внутренняя дилатация исключает свечение самого ромба из фона.
        inner = cv2.dilate(
            core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        ring_size = 2 * ring_radius + 1
        outer = cv2.dilate(
            core,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ring_size, ring_size)
            ),
        )
        ring = (outer > 0) & (inner == 0)
        core_mask = core > 0
        if np.count_nonzero(core_mask) < 3 or np.count_nonzero(ring) < 12:
            return False

        day_crop = day_normalized[y1:y2, x1:x2]
        night_crop = night_normalized[y1:y2, x1:x2]
        day_values = day_crop[core_mask]
        night_values = night_crop[core_mask]
        aligned_gain = night_values - day_values

        # Медиана устойчива к одному пересвеченному лучу ромба и к остаточному
        # субпиксельному сдвигу Day/Night после выравнивания кадров.
        day_core = float(np.median(day_values))
        day_ring = float(np.median(day_crop[ring]))
        night_core = float(np.median(night_values))
        night_ring = float(np.median(night_crop[ring]))
        core_gain = float(np.median(aligned_gain))
        core_gain_high = float(np.percentile(aligned_gain, 75.0))
        day_contrast = day_core - day_ring
        night_contrast = night_core - night_ring
        contrast_gain = night_contrast - day_contrast

        candidate["day_core"] = day_core
        candidate["day_ring"] = day_ring
        candidate["night_core"] = night_core
        candidate["night_ring"] = night_ring
        candidate["day_local_contrast"] = day_contrast
        candidate["night_local_contrast"] = night_contrast
        candidate["local_contrast_gain"] = contrast_gain
        candidate["core_gain"] = core_gain
        candidate["core_gain_high"] = core_gain_high
        candidate["core_pixel_count"] = int(pixels.shape[0])

        # Решающий тест выполняется в ИСХОДНОЙ яркости, а не только после
        # нормализации всего кадра. Нормализация способна сделать темную
        # статичную текстуру «яркой» относительно нового диапазона и именно
        # так ранее был ошибочно принят верхний ложный P2.
        raw_cycle_count = min(
            len(day_cycles_raw or []), len(night_cycles_raw or [])
        )
        raw_day_levels = []
        raw_night_levels = []
        raw_night_high_levels = []
        raw_cycle_gains = []
        raw_day_positions = []
        raw_day_black_fractions = []

        def image_scale(image: np.ndarray) -> float:
            return 255.0 if float(np.max(image)) > 2.0 else 1.0

        day_match_radius = int(
            round(max(7.0, min(24.0, object_size * 1.50 + 5.0)))
        )
        black_limit = float(np.clip(day_black_threshold, 10, 160)) / 255.0

        def match_day_black_core(image: np.ndarray):
            """Ищет черную копию Night-ядра независимо от его точного сдвига."""
            scale = image_scale(image)
            best = None
            for shift_y in range(-day_match_radius, day_match_radius + 1):
                for shift_x in range(-day_match_radius, day_match_radius + 1):
                    distance = float(np.hypot(shift_x, shift_y))
                    if distance > day_match_radius:
                        continue
                    shifted_x = pixels[:, 0] + shift_x
                    shifted_y = pixels[:, 1] + shift_y
                    valid = (
                        (shifted_x >= 0)
                        & (shifted_x < image.shape[1])
                        & (shifted_y >= 0)
                        & (shifted_y < image.shape[0])
                    )
                    if np.count_nonzero(valid) < max(3, int(0.85 * len(pixels))):
                        continue
                    values = (
                        image[shifted_y[valid], shifted_x[valid]] / scale
                    )
                    level = float(np.median(values))
                    black_fraction = float(np.mean(values <= black_limit))
                    if black_fraction < 0.62 or level > black_limit:
                        continue
                    distance_score = float(
                        np.exp(
                            -(
                                distance
                                / max(3.0, day_match_radius * 0.58)
                            )
                            ** 2
                        )
                    )
                    darkness_score = float(
                        np.clip((black_limit - level) / max(0.08, black_limit), 0.0, 1.0)
                    )
                    score = (
                        0.62 * distance_score
                        + 0.28 * black_fraction
                        + 0.10 * darkness_score
                    )
                    if best is None or score > best[0]:
                        best = (
                            score,
                            level,
                            black_fraction,
                            (
                                float(candidate["position"][0] + shift_x),
                                float(candidate["position"][1] + shift_y),
                            ),
                        )
            return best

        for cycle_index in range(raw_cycle_count):
            day_image = np.asarray(day_cycles_raw[cycle_index])
            night_image = np.asarray(night_cycles_raw[cycle_index])
            if day_image.shape != day_normalized.shape or night_image.shape != day_normalized.shape:
                continue
            day_match = match_day_black_core(day_image)
            if day_match is None:
                continue
            _, day_level, black_fraction, day_position = day_match
            night_scale = image_scale(night_image)
            night_cycle_values = (
                night_image[pixels[:, 1], pixels[:, 0]] / night_scale
            )
            night_level = float(np.median(night_cycle_values))
            night_high = float(np.percentile(night_cycle_values, 75.0))
            raw_day_levels.append(day_level)
            raw_night_levels.append(night_level)
            raw_night_high_levels.append(night_high)
            raw_cycle_gains.append(night_level - day_level)
            raw_day_positions.append(day_position)
            raw_day_black_fractions.append(black_fraction)

        if raw_cycle_count >= 2 and len(raw_cycle_gains) < 2:
            candidate["signature_failure"] = "SIG_NOT_BLACK"
            candidate["day_black_match_radius"] = int(day_match_radius)
            return False
        has_repeated_raw_check = len(raw_cycle_gains) >= 2
        if has_repeated_raw_check:
            raw_day_level = float(max(raw_day_levels))
            raw_night_level = float(min(raw_night_levels))
            raw_night_high = float(min(raw_night_high_levels))
            raw_gain_min = float(min(raw_cycle_gains))
            candidate["raw_day_level"] = raw_day_level
            candidate["raw_night_level"] = raw_night_level
            candidate["raw_night_high"] = raw_night_high
            candidate["raw_cycle_gains"] = [
                float(value) for value in raw_cycle_gains
            ]
            candidate["raw_gain_min"] = raw_gain_min
            candidate["day_black_positions"] = [
                [float(position[0]), float(position[1])]
                for position in raw_day_positions
            ]
            candidate["day_black_fraction_min"] = float(
                min(raw_day_black_fractions)
            )
            candidate["day_black_match_radius"] = int(day_match_radius)

            day_position_spread = float(
                np.linalg.norm(
                    np.asarray(raw_day_positions[0])
                    - np.asarray(raw_day_positions[1])
                )
            )
            candidate["day_black_position_spread"] = day_position_spread
            if day_position_spread > max(6.0, day_match_radius * 0.40):
                candidate["signature_failure"] = "SIG_DAY_UNSTABLE"
                return False

            # Оба Night должны показать реально яркое ядро. Темная деталь,
            # получившая высокий балл лишь после нормализации, здесь отпадает.
            if raw_night_level < 0.46 or raw_night_high < 0.58:
                candidate["signature_failure"] = "SIG_NIGHT"
                return False
            # В каждом Day в пределах ошибки совмещения обязано находиться
            # компактное ЧЕРНОЕ ядро той же формы. Светлая поверхность рядом
            # больше не участвует в оценке P2.
            if raw_day_level > black_limit:
                candidate["signature_failure"] = "SIG_NOT_BLACK"
                return False
            # В каждом из двух циклов обязан присутствовать физический
            # прирост яркости от черного Day к яркому Night.
            if raw_gain_min < 0.14:
                candidate["signature_failure"] = "SIG_STATIC"
                return False
        else:
            # Совместимость со старыми тестами/вызовами. Живой поиск v5.23
            # всегда передает два полных цикла исходных кадров.
            if night_core < 0.45:
                candidate["signature_failure"] = "SIG_NIGHT"
                return False
            if day_core > 0.82:
                candidate["signature_failure"] = "SIG_DAY"
                return False
            if core_gain < 0.12 and core_gain_high < 0.18:
                candidate["signature_failure"] = "SIG_GAIN"
                return False

        strong_core_transition = (
            candidate.get("raw_gain_min", -np.inf) >= 0.20
            if has_repeated_raw_check
            else (core_gain >= 0.23 or core_gain_high >= 0.30)
        )
        local_transition = (
            night_contrast >= 0.055 and contrast_gain >= 0.075
        )
        if not (strong_core_transition or local_transition):
            candidate["signature_failure"] = "SIG_LOCAL"
            return False

        signature_score = float(
            np.clip(
                0.28 * np.clip(core_gain_high / 0.60, 0.0, 1.0)
                + 0.18 * np.clip(night_core / 0.95, 0.0, 1.0)
                + 0.12 * np.clip(night_contrast / 0.45, 0.0, 1.0)
                + 0.10 * np.clip(contrast_gain / 0.45, 0.0, 1.0)
                + 0.16
                * np.clip(candidate.get("raw_gain_min", core_gain) / 0.55, 0.0, 1.0)
                + 0.10
                * np.clip(candidate.get("raw_night_level", night_core) / 0.95, 0.0, 1.0)
                + 0.06
                * np.clip(
                    candidate.get("tracking_diamond_score", 0.0), 0.0, 1.0
                ),
                0.0,
                1.0,
            )
        )
        candidate["signature_score"] = signature_score
        candidate["quality"] = float(
            np.clip(
                0.42 * float(candidate.get("quality", 0.0))
                + 0.58 * signature_score,
                0.0,
                1.0,
            )
        )
        candidate["signature_failure"] = ""
        return True

    def _analyze_ir_flash_frames(
        self,
        day_samples: List[np.ndarray],
        night_samples: List[np.ndarray],
        day_verify_samples: Optional[List[np.ndarray]] = None,
        night_verify_samples: Optional[List[np.ndarray]] = None,
    ):
        if not all(
            (
                day_samples,
                night_samples,
                day_verify_samples,
                night_verify_samples,
            )
        ):
            raise RuntimeError(
                "Не получены два полных цикла Day→Night; результат не принимается"
            )

        day_first = np.median(np.stack(day_samples), axis=0).astype(np.float32)
        night_first = np.median(np.stack(night_samples), axis=0).astype(np.float32)
        day_second = np.median(
            np.stack(day_verify_samples), axis=0
        ).astype(np.float32)
        night_second = np.median(
            np.stack(night_verify_samples), axis=0
        ).astype(np.float32)
        shapes = {
            day_first.shape,
            night_first.shape,
            day_second.shape,
            night_second.shape,
        }
        if len(shapes) != 1:
            raise RuntimeError("Размер кадра изменился при переключении Day/Night")

        # Все четыре состояния приводятся к координатам первого Night. Это
        # позволяет потребовать повторную вспышку строго в том же месте.
        day_first, alignment = self._align_ir_day_frame(day_first, night_first)
        day_second, alignment_day_second = self._align_ir_day_frame(
            day_second, night_first
        )
        night_second, alignment_night_second = self._align_ir_day_frame(
            night_second, night_first
        )
        raw_day_cycles = [day_first, day_second]
        raw_night_cycles = [night_first, night_second]
        day = np.median(np.stack(raw_day_cycles), axis=0).astype(np.float32)
        night = np.median(np.stack(raw_night_cycles), axis=0).astype(np.float32)

        # Сравниваем не абсолютные уровни, а фотометрически нормированный
        # ИК-прирост. Белый объект, светлый в обоих режимах, получает почти
        # нулевой вес. Темная в Day и яркая в Night ретрометка — высокий вес.
        def robust_normalize(image):
            low, high = np.percentile(image, (5.0, 98.5))
            span = max(20.0, float(high - low))
            return np.clip((image - low) / span, 0.0, 1.35)

        day_cycle_normalized = [robust_normalize(image) for image in raw_day_cycles]
        night_cycle_normalized = [
            robust_normalize(image) for image in raw_night_cycles
        ]

        def flash_response(day_image, night_image, day_raw, night_raw):
            day_local = day_image - cv2.GaussianBlur(
                day_image, (0, 0), 15.0
            )
            night_local = night_image - cv2.GaussianBlur(
                night_image, (0, 0), 15.0
            )
            absolute_gain = np.maximum(night_image - day_image, 0.0)
            relative_gain = np.maximum(
                np.log((night_image + 0.06) / (day_image + 0.06)),
                0.0,
            )
            local_gain = np.maximum(night_local - day_local, 0.0)
            day_dark_weight = np.clip((0.84 - day_image) / 0.58, 0.0, 1.0)
            night_bright_weight = np.clip(
                (night_image - 0.28) / 0.55, 0.0, 1.0
            )
            current = (
                85.0 * absolute_gain
                + 52.0 * relative_gain
                + 95.0 * local_gain
            ) * day_dark_weight * night_bright_weight
            # Второй независимый канал использует физическую яркость 0…255.
            # Он не зависит от того, как изменилась гистограмма всего кадра.
            day_raw_unit = np.clip(day_raw / 255.0, 0.0, 1.0)
            night_raw_unit = np.clip(night_raw / 255.0, 0.0, 1.0)
            raw_gain = np.maximum(night_raw_unit - day_raw_unit, 0.0)
            black_limit = (
                float(self.calibration.ir_day_black_threshold) / 255.0
            )
            raw_day_dark = np.clip(
                (black_limit + 0.10 - day_raw_unit) / 0.18,
                0.0,
                1.0,
            )
            raw_night_bright = np.clip(
                (night_raw_unit - 0.42) / 0.45, 0.0, 1.0
            )
            current += (
                210.0 * raw_gain * raw_day_dark * raw_night_bright
            )
            return cv2.GaussianBlur(
                current.astype(np.float32), (0, 0), 0.8
            )

        response_first = flash_response(
            day_cycle_normalized[0],
            night_cycle_normalized[0],
            raw_day_cycles[0],
            raw_night_cycles[0],
        )
        response_second = flash_response(
            day_cycle_normalized[1],
            night_cycle_normalized[1],
            raw_day_cycles[1],
            raw_night_cycles[1],
        )
        # Не среднее и не максимум: кандидат обязан дать положительный отклик
        # в ОБОИХ независимых переключениях. Разовый блик и статичная текстура
        # с ошибкой экспозиции не получают высокий response.
        response = np.minimum(response_first, response_second)
        day_normalized = np.maximum(
            day_cycle_normalized[0], day_cycle_normalized[1]
        )
        night_normalized = np.minimum(
            night_cycle_normalized[0], night_cycle_normalized[1]
        )

        # v5.8: сначала ищем ромбовидный Day→Night отклик отдельно возле
        # каждой РУЧНОЙ исходной области. Пропущенные точки получают резервные
        # кандидаты из всего кадра. Ошибочно сдвинутая область прошлой версии
        # больше не мешает найти фактический P2 справа.
        strict_regions = bool(self.calibration.ir_strict_regions)
        configured_scale = max(1.0, float(self.calibration.ir_search_scale))
        search_scale = min(configured_scale, 1.5) if strict_regions else configured_scale
        reference_regions = (
            list(self.calibration.ir_reference_regions)
            if len(self.calibration.ir_reference_regions)
            == len(self.calibration.peak_regions)
            else list(self.calibration.peak_regions)
        )
        raw_candidates = []
        local_thresholds = []
        search_rects = []
        proposed_search_regions = []
        for region in reference_regions:
            x, y, width, height = region
            region_center = np.array(
                [x + width / 2.0, y + height / 2.0], dtype=np.float32
            )
            expanded = clip_rect(
                (
                    int(round(region_center[0] - width * search_scale / 2.0)),
                    int(round(region_center[1] - height * search_scale / 2.0)),
                    max(3, int(round(width * search_scale))),
                    max(3, int(round(height * search_scale))),
                ),
                response.shape,
            )
            proposed_search_regions.append(expanded or region)

        exclusive_search_regions = make_regions_exclusive(
            proposed_search_regions, reference_regions, response.shape
        )
        for region_index, expanded in enumerate(exclusive_search_regions):
            search_rects.append((region_index, expanded))
            individual = self.calibration.region_settings.get(
                str(region_index + 1), {}
            )
            region_candidates, local_threshold = self._extract_ir_candidates(
                response,
                day_normalized,
                night_normalized,
                night,
                expanded,
                detection_settings=individual,
                region_hint=region_index,
                source="local",
            )
            local_thresholds.append(local_threshold)
            raw_candidates.extend(region_candidates[:8])

        if self.calibration.ir_global_fallback:
            global_candidates, global_threshold = self._extract_ir_candidates(
                response,
                day_normalized,
                night_normalized,
                night,
                (0, 0, response.shape[1], response.shape[0]),
                detection_settings=None,
                region_hint=None,
                source="global",
            )
            local_thresholds.append(global_threshold)
            raw_candidates.extend(
                global_candidates[: max(24, len(reference_regions) * 10)]
            )

        # Объединяем один и тот же ореол, найденный локальным и глобальным
        # проходами. Сохраняем все допустимые Pn-подсказки.
        candidates = []
        for candidate in sorted(
            raw_candidates,
            key=lambda item: (item["quality"], item["strength"]),
            reverse=True,
        ):
            duplicate = None
            for existing in candidates:
                distance = float(
                    np.linalg.norm(
                        np.asarray(candidate["position"])
                        - np.asarray(existing["position"])
                    )
                )
                merge_distance = max(
                    6.0,
                    0.25
                    * min(
                        max(candidate["bbox"][2:]),
                        max(existing["bbox"][2:]),
                    ),
                )
                if distance <= merge_distance:
                    duplicate = existing
                    break
            if duplicate is None:
                candidates.append(candidate)
            else:
                duplicate["region_hints"].update(candidate["region_hints"])
                if duplicate["source"] != candidate["source"]:
                    duplicate["source"] = "local+global"
        # Каждый кандидат независимо обязан пройти второй детектор по Night и
        # локальную проверку «темное ядро Day → яркое ядро Night». Только после
        # этого разрешается назначать ему номер Pn. Это исключает однотипный
        # выбор текстуры крыши/откоса вместо отражателя.
        night_u8 = np.clip(night, 0, 255).astype(np.uint8)
        validated_candidates = []
        rejected_candidates = []
        for candidate in candidates:
            hints = set(candidate.get("region_hints", set()))
            eligible_regions = (
                sorted(hints)
                if hints
                else list(range(len(reference_regions)))
            )
            if not eligible_regions:
                continue
            candidate_position = np.asarray(candidate["position"])
            refinement_region = min(
                eligible_regions,
                key=lambda index: float(
                    np.linalg.norm(
                        candidate_position
                        - np.asarray(
                            [
                                reference_regions[index][0]
                                + reference_regions[index][2] / 2.0,
                                reference_regions[index][1]
                                + reference_regions[index][3] / 2.0,
                            ]
                        )
                    )
                ),
            )
            if not self._refine_ir_candidate_on_night(
                night_u8, candidate, refinement_region
            ):
                candidate["rejection_reason"] = "NIGHT_CORE"
                rejected_candidates.append(candidate)
                continue
            if not self._validate_ir_local_signature(
                day_normalized,
                night_normalized,
                candidate,
                day_cycles_raw=raw_day_cycles,
                night_cycles_raw=raw_night_cycles,
                day_black_threshold=self.calibration.ir_day_black_threshold,
            ):
                candidate["rejection_reason"] = candidate.get(
                    "signature_failure", "DAY_NIGHT_SIGNATURE"
                )
                rejected_candidates.append(candidate)
                continue
            validated_candidates.append(candidate)

        # После уточнения два response-фрагмента могут попасть в одно и то же
        # Night-ядро. Оставляем один, объединяя сведения о локальных областях.
        candidates = []
        for candidate in sorted(
            validated_candidates,
            key=lambda item: (item["quality"], item["strength"]),
            reverse=True,
        ):
            duplicate = None
            for existing in candidates:
                distance = float(
                    np.linalg.norm(
                        np.asarray(candidate["position"])
                        - np.asarray(existing["position"])
                    )
                )
                if distance <= max(
                    4.0,
                    0.25
                    * min(
                        max(candidate["tracking_bbox"][2:]),
                        max(existing["tracking_bbox"][2:]),
                    ),
                ):
                    duplicate = existing
                    break
            if duplicate is None:
                candidates.append(candidate)
            else:
                duplicate["region_hints"].update(candidate["region_hints"])
                if duplicate["source"] != candidate["source"]:
                    duplicate["source"] = "local+global"
        candidates = candidates[: max(30, len(reference_regions) * 12)]

        # Положение внутри назначенной пользователем области теперь важнее
        # небольшой разницы яркости между несколькими допустимыми ромбами.
        # Назначение выполняется глобально, а не жадно по одной лучшей паре.
        matches = assign_ir_candidates_to_regions(reference_regions, candidates)

        threshold = (
            float(np.median(local_thresholds))
            if local_thresholds
            else float(self.calibration.ir_flash_delta)
        )

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
        for region_index, (x, y, width, height) in search_rects:
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                (255, 190, 0),
                1,
            )
            cv2.putText(
                preview,
                f"P{region_index + 1}",
                (x + 2, max(14, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 190, 0),
                1,
                cv2.LINE_AA,
            )
        for candidate in rejected_candidates:
            x, y, width, height = candidate["bbox"]
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                (0, 0, 200),
                1,
            )
            cv2.putText(
                preview,
                f"X:{candidate.get('rejection_reason', 'FILTER')}",
                (x, max(14, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
        for candidate in candidates:
            x, y, width, height = candidate["bbox"]
            candidate_color = (
                (210, 0, 210)
                if candidate.get("source") == "global"
                else (0, 165, 255)
            )
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                candidate_color,
                1,
            )
            cv2.putText(
                preview,
                (
                    f"D:{candidate['diamond_score']:.2f} "
                    f"S:{candidate.get('signature_score', 0.0):.2f}"
                ),
                (x, min(preview.shape[0] - 4, y + height + 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                candidate_color,
                1,
                cv2.LINE_AA,
            )
            allowed_routes = [
                (int(region_id), route)
                for region_id, route in candidate.get(
                    "assignment_routes", {}
                ).items()
                if route.get("allowed", False)
            ]
            if allowed_routes:
                route_region, best_route = max(
                    allowed_routes,
                    key=lambda item: float(
                        item[1].get("assignment_score", 0.0)
                    ),
                )
                route_text = (
                    f"R:P{route_region} "
                    f"L:{best_route.get('distance_px', 0.0):.0f} "
                    f"W:{best_route.get('line_weight', 0.0):.2f}"
                )
            else:
                route_text = "R:BLOCK"
            cv2.putText(
                preview,
                route_text,
                (x, min(preview.shape[0] - 4, y + height + 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                candidate_color,
                1,
                cv2.LINE_AA,
            )
        for region_index, candidate in matches:
            px, py = [int(round(value)) for value in candidate["position"]]
            origin = candidate.get("assignment_origin")
            if not origin:
                region_x, region_y, region_width, region_height = (
                    reference_regions[region_index]
                )
                origin = [
                    region_x + region_width / 2.0,
                    region_y + region_height / 2.0,
                ]
            origin_point = tuple(int(round(value)) for value in origin)
            cv2.line(
                preview,
                origin_point,
                (px, py),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                preview,
                origin_point,
                (0, 255, 255),
                cv2.MARKER_CROSS,
                9,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(preview, (px, py), 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                preview,
                (
                    f"P{region_index + 1} IR "
                    f"S:{candidate.get('signature_score', 0.0):.2f}"
                ),
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
                    f"L:{candidate.get('assignment_distance_px', 0.0):.1f} "
                    f"W:{candidate.get('assignment_line_weight', 0.0):.2f}"
                ),
                (px + 12, py + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            preview,
            (
                f"IR x2 verified: {len(matches)}/{len(self.calibration.peak_regions)}; "
                f"T={threshold:.1f}; "
                f"D1=({alignment[0]:+.1f},{alignment[1]:+.1f}); "
                f"D2=({alignment_day_second[0]:+.1f},{alignment_day_second[1]:+.1f}); "
                f"N2=({alignment_night_second[0]:+.1f},{alignment_night_second[1]:+.1f})"
            ),
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return matches, candidates, rejected_candidates, preview, threshold

    @staticmethod
    def _write_png_unicode(path: Path, image: np.ndarray):
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError(f"Не удалось закодировать {path.name}")
        encoded.tofile(str(path))

    def _save_ir_scan_diagnostics(
        self, preview, matches, candidates, rejected_candidates, threshold
    ) -> Optional[Path]:
        """Автоматически сохраняет Day, Night, результат и координаты поиска."""
        try:
            output_dir = Path(__file__).resolve().parent / "ir_scan_records"
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base = output_dir / f"ir_scan_{stamp}"
            if self._ir_day_samples:
                day = np.median(np.stack(self._ir_day_samples), axis=0).astype(
                    np.uint8
                )
                self._write_png_unicode(
                    base.with_name(base.name + "_day.png"), day
                )
            if self._ir_night_samples:
                night = np.median(np.stack(self._ir_night_samples), axis=0).astype(
                    np.uint8
                )
                self._write_png_unicode(
                    base.with_name(base.name + "_night.png"), night
                )
            if self._ir_day_verify_samples:
                day_verify = np.median(
                    np.stack(self._ir_day_verify_samples), axis=0
                ).astype(np.uint8)
                self._write_png_unicode(
                    base.with_name(base.name + "_day2.png"), day_verify
                )
            if self._ir_night_verify_samples:
                night_verify = np.median(
                    np.stack(self._ir_night_verify_samples), axis=0
                ).astype(np.uint8)
                self._write_png_unicode(
                    base.with_name(base.name + "_night2.png"), night_verify
                )
            result_path = base.with_name(base.name + "_result.png")
            self._write_png_unicode(result_path, preview)
            metadata = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "threshold": float(threshold),
                "expected": len(self.calibration.peak_regions),
                "found": len(matches),
                "points": {
                    f"P{region_index + 1}": {
                        "position": [
                            float(candidate["position"][0]),
                            float(candidate["position"][1]),
                        ],
                        "bbox": [int(value) for value in candidate["bbox"]],
                        "area": float(candidate["area"]),
                        "diamond_score": float(candidate["diamond_score"]),
                        "quality": float(candidate["quality"]),
                        "response_position": [
                            float(value)
                            for value in candidate.get(
                                "response_position", candidate["position"]
                            )
                        ],
                        "center_correction": float(
                            candidate.get("center_correction", 0.0)
                        ),
                        "tracking_area": float(
                            candidate.get("tracking_area", 0.0)
                        ),
                        "signature_score": float(
                            candidate.get("signature_score", 0.0)
                        ),
                        "day_core": float(candidate.get("day_core", 0.0)),
                        "night_core": float(candidate.get("night_core", 0.0)),
                        "day_local_contrast": float(
                            candidate.get("day_local_contrast", 0.0)
                        ),
                        "night_local_contrast": float(
                            candidate.get("night_local_contrast", 0.0)
                        ),
                        "local_contrast_gain": float(
                            candidate.get("local_contrast_gain", 0.0)
                        ),
                        "core_gain": float(candidate.get("core_gain", 0.0)),
                        "core_gain_high": float(
                            candidate.get("core_gain_high", 0.0)
                        ),
                        "core_pixel_count": int(
                            candidate.get("core_pixel_count", 0)
                        ),
                        "raw_day_level": float(
                            candidate.get("raw_day_level", 0.0)
                        ),
                        "raw_night_level": float(
                            candidate.get("raw_night_level", 0.0)
                        ),
                        "raw_gain_min": float(
                            candidate.get("raw_gain_min", 0.0)
                        ),
                        "raw_cycle_gains": [
                            float(value)
                            for value in candidate.get("raw_cycle_gains", [])
                        ],
                        "day_black_positions": candidate.get(
                            "day_black_positions", []
                        ),
                        "day_black_fraction_min": float(
                            candidate.get("day_black_fraction_min", 0.0)
                        ),
                        "day_black_position_spread": float(
                            candidate.get("day_black_position_spread", 0.0)
                        ),
                        "day_black_match_radius": int(
                            candidate.get("day_black_match_radius", 0)
                        ),
                        "assignment_origin": [
                            float(value)
                            for value in candidate.get(
                                "assignment_origin", (0.0, 0.0)
                            )
                        ],
                        "assignment_distance_px": float(
                            candidate.get("assignment_distance_px", 0.0)
                        ),
                        "assignment_line_weight": float(
                            candidate.get("assignment_line_weight", 0.0)
                        ),
                        "assignment_score": float(
                            candidate.get("assignment_score", 0.0)
                        ),
                    }
                    for region_index, candidate in matches
                },
                "candidates": [
                    {
                        "position": [
                            float(candidate["position"][0]),
                            float(candidate["position"][1]),
                        ],
                        "bbox": [int(value) for value in candidate["bbox"]],
                        "diamond_score": float(candidate["diamond_score"]),
                        "signature_score": float(
                            candidate.get("signature_score", 0.0)
                        ),
                        "quality": float(candidate.get("quality", 0.0)),
                        "source": str(candidate.get("source", "")),
                        "assignment_routes": candidate.get(
                            "assignment_routes", {}
                        ),
                    }
                    for candidate in candidates
                ],
                "rejected_candidates": [
                    {
                        "position": [
                            float(candidate["position"][0]),
                            float(candidate["position"][1]),
                        ],
                        "bbox": [int(value) for value in candidate["bbox"]],
                        "reason": str(
                            candidate.get("rejection_reason", "FILTER")
                        ),
                        "diamond_score": float(
                            candidate.get("diamond_score", 0.0)
                        ),
                        "quality": float(candidate.get("quality", 0.0)),
                        "day_core": float(candidate.get("day_core", 0.0)),
                        "night_core": float(
                            candidate.get("night_core", 0.0)
                        ),
                        "day_local_contrast": float(
                            candidate.get("day_local_contrast", 0.0)
                        ),
                        "night_local_contrast": float(
                            candidate.get("night_local_contrast", 0.0)
                        ),
                        "local_contrast_gain": float(
                            candidate.get("local_contrast_gain", 0.0)
                        ),
                        "core_gain": float(candidate.get("core_gain", 0.0)),
                        "core_gain_high": float(
                            candidate.get("core_gain_high", 0.0)
                        ),
                        "core_pixel_count": int(
                            candidate.get("core_pixel_count", 0)
                        ),
                        "raw_day_level": float(
                            candidate.get("raw_day_level", 0.0)
                        ),
                        "raw_night_level": float(
                            candidate.get("raw_night_level", 0.0)
                        ),
                        "raw_gain_min": float(
                            candidate.get("raw_gain_min", 0.0)
                        ),
                        "raw_cycle_gains": [
                            float(value)
                            for value in candidate.get("raw_cycle_gains", [])
                        ],
                        "day_black_positions": candidate.get(
                            "day_black_positions", []
                        ),
                        "day_black_fraction_min": float(
                            candidate.get("day_black_fraction_min", 0.0)
                        ),
                        "day_black_position_spread": float(
                            candidate.get("day_black_position_spread", 0.0)
                        ),
                        "day_black_match_radius": int(
                            candidate.get("day_black_match_radius", 0)
                        ),
                    }
                    for candidate in rejected_candidates
                ],
                "settings": {
                    "ir_flash_delta": self.calibration.ir_flash_delta,
                    "ir_day_black_threshold": (
                        self.calibration.ir_day_black_threshold
                    ),
                    "ir_diamond_min_score": self.calibration.ir_diamond_min_score,
                    "ir_lock_radius": self.calibration.ir_lock_radius,
                    "ir_max_travel": self.calibration.ir_max_travel,
                    "ir_search_scale": self.calibration.ir_search_scale,
                },
            }
            with open(
                base.with_name(base.name + "_data.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
            return result_path
        except Exception as exc:
            logger.warning("Не удалось сохранить диагностику ИК-поиска: %s", exc)
            return None

    def _initialize_ir_tracking_models(self):
        """Калибрует Night-маску тем же детектором, который ведет треки."""
        if not self._ir_night_samples:
            return
        night_gray = np.median(
            np.stack(self._ir_night_samples), axis=0
        ).astype(np.uint8)
        night_frame = cv2.cvtColor(night_gray, cv2.COLOR_GRAY2BGR)
        detector = ReflectorDetector(self.calibration)
        anchor_radius = max(3.0, float(self.calibration.ir_lock_radius))
        proposed_regions = []
        for index, region in enumerate(self.calibration.peak_regions):
            x, y, width, height = region
            center = (x + width / 2.0, y + height / 2.0)
            saved_anchor = self.calibration.ir_confirmed_centers.get(
                str(index + 1)
            )
            scale = 1.0
            if saved_anchor and len(saved_anchor) == 2:
                margin = anchor_radius + 3.0
                scale = max(
                    1.0,
                    2.0 * (abs(float(saved_anchor[0]) - center[0]) + margin)
                    / width,
                    2.0 * (abs(float(saved_anchor[1]) - center[1]) + margin)
                    / height,
                )
                scale = min(
                    max(1.0, float(self.calibration.roi_max_scale)), scale
                )
            proposed_regions.append(
                ReflectorTracker._centered_region(
                    center, region, scale, night_frame.shape
                )
            )
        model_regions = make_regions_exclusive(
            proposed_regions,
            list(self.calibration.peak_regions),
            night_frame.shape,
        )
        for index, region in enumerate(self.calibration.peak_regions):
            point_id = str(index + 1)
            saved_anchor = self.calibration.ir_confirmed_centers.get(point_id)
            model = self.calibration.ir_confirmed_models.get(point_id)
            if not saved_anchor or len(saved_anchor) != 2 or model is None:
                continue
            anchor = (float(saved_anchor[0]), float(saved_anchor[1]))
            detection_region = (
                model_regions[index] if index < len(model_regions) else region
            )
            detection = detector.detect_in_region(
                night_frame,
                detection_region,
                predicted=anchor,
                detection_settings=self.calibration.region_settings.get(point_id),
                anchor=anchor,
                anchor_radius=anchor_radius,
                anchor_model=model,
            )
            if detection is None:
                continue
            center_offset = (
                np.asarray(detection.position, dtype=np.float64)
                - np.asarray(anchor, dtype=np.float64)
            )
            maximum_offset = max(2.0, min(6.0, anchor_radius * 0.75))
            if (
                not np.all(np.isfinite(center_offset))
                or float(np.linalg.norm(center_offset)) > maximum_offset
            ):
                # Большая разница означает, что обычный Night-детектор взял
                # соседний объект, а не то же ядро. Такой сдвиг нельзя
                # превращать в допустимую калибровочную поправку.
                model["tracking_center_offset_x"] = 0.0
                model["tracking_center_offset_y"] = 0.0
                continue
            model["tracking_area"] = float(detection.area)
            model["tracking_diamond_score"] = float(detection.diamond_score)
            model["tracking_radius"] = float(detection.radius)
            model["tracking_center_offset_x"] = float(center_offset[0])
            model["tracking_center_offset_y"] = float(center_offset[1])

    def _finish_ir_flash_scan(self, token: int, result):
        if token != self._ir_scan_token:
            return
        matches, candidates, rejected_candidates, preview, threshold = result
        confirmed_centers = {
            str(region_index + 1): [
                float(candidate["position"][0]),
                float(candidate["position"][1]),
            ]
            for region_index, candidate in matches
        }
        self.calibration.ir_confirmed_centers = confirmed_centers
        self.calibration.ir_confirmed_models = {
            str(region_index + 1): {
                "area": float(candidate["area"]),
                "bbox": [int(value) for value in candidate["bbox"]],
                "diamond_score": float(candidate["diamond_score"]),
                "quality": float(candidate["quality"]),
                "response": float(candidate["response"]),
                "night_peak": float(candidate.get("night_peak", 0.0)),
                "halo_radius": float(candidate["halo_radius"]),
                "tracking_area": float(candidate.get("tracking_area", 0.0)),
                "tracking_diamond_score": float(
                    candidate.get("tracking_diamond_score", 0.0)
                ),
                "tracking_radius": float(
                    candidate.get("tracking_radius", 0.0)
                ),
                "tracking_center_offset_x": 0.0,
                "tracking_center_offset_y": 0.0,
            }
            for region_index, candidate in matches
        }
        self.calibration.ir_verification_active = True
        self.calibration.ir_model_version = 10
        if matches:
            # ИК-поиск сохраняет подтвержденные центры отдельно и больше не
            # переносит пользовательские области к найденным точкам. Поэтому
            # исходные непересекающиеся территории P1…Pn остаются неизменными.
            if len(self.calibration.ir_reference_regions) == len(
                self.calibration.peak_regions
            ):
                self.calibration.peak_regions = list(
                    self.calibration.ir_reference_regions
                )
                regions = self.calibration.peak_regions
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
            # В момент успешного ИК-поиска Night-кадры уже доступны. Сразу
            # создаем сопоставимую модель обычного сопровождения, чтобы первый
            # следующий кадр не сравнивался с response-маской другого типа.
            self._initialize_ir_tracking_models()
        if self.tracker:
            self._reset_processing_modules(announce=False)
        self._update_region_status()

        self._ir_preview_overlay = preview
        self._ir_preview_until = time.time() + 6.0
        diagnostic_path = self._save_ir_scan_diagnostics(
            preview,
            matches,
            candidates,
            rejected_candidates,
            threshold,
        )
        self._ir_scan_running = False
        self._ir_day_samples = []
        self._ir_night_samples = []
        self._ir_day_verify_samples = []
        self._ir_night_verify_samples = []
        self.ir_scan_btn.config(
            state=tk.NORMAL, text="🔦 Найти отражатели: Day ↔ Night ×2"
        )
        found = len(matches)
        expected = len(self.calibration.peak_regions)
        color = "green" if found == expected else "#b05a00"
        self.hik_status_label.config(
            text=(
                f"ИК-поиск: найдено {found}/{expected}; порог {threshold:.1f}. "
                "Камера оставлена в Night. "
                + (
                    f"Снимок: {diagnostic_path.name}."
                    if diagnostic_path is not None
                    else "Не удалось сохранить снимок."
                )
            ),
            foreground=color,
        )
        if found < expected:
            found_ids = {region_index for region_index, _ in matches}
            missing = ", ".join(
                f"P{index + 1}"
                for index in range(expected)
                if index not in found_ids
            )
            self.status_label.config(
                text=(
                    f"Не подтверждены: {missing}. Для них обычный яркий "
                    "объект не получит LOCK; повторите Day↔Night ×2 после проверки областей."
                )
            )
        else:
            self.status_label.config(
                text=(
                    "Все ИК-отражатели подтверждены; исходные области "
                    "сохранены без пересечений, поиск начат заново"
                )
            )

    def _abort_ir_flash_scan(self, error, restore_mode: bool):
        original_mode = self._ir_original_mode
        self._ir_scan_token += 1
        self._ir_scan_running = False
        self._ir_day_samples = []
        self._ir_night_samples = []
        self._ir_day_verify_samples = []
        self._ir_night_verify_samples = []
        self.ir_scan_btn.config(
            state=tk.NORMAL, text="🔦 Найти отражатели: Day ↔ Night ×2"
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
            day_black_threshold = int(
                round(float(self.ir_day_black_var.get()))
            )
            search_scale = float(
                str(self.ir_search_scale_var.get()).replace(",", ".")
            )
            lock_radius = float(
                str(self.ir_lock_radius_var.get()).replace(",", ".")
            )
            max_travel = float(
                str(self.ir_max_travel_var.get()).replace(",", ".")
            )
            diamond_min_score = float(
                str(self.ir_diamond_min_score_var.get()).replace(",", ".")
            )
            sample_count = int(self.calibration.ir_sample_count)
        except (tk.TclError, ValueError) as exc:
            raise ValueError("Проверьте числовые параметры ИК-поиска") from exc
        self.calibration.ir_settle_seconds = float(np.clip(settle, 0.5, 10.0))
        self.calibration.ir_flash_delta = int(np.clip(flash_delta, 5, 200))
        self.calibration.ir_day_black_threshold = int(
            np.clip(day_black_threshold, 10, 160)
        )
        self.calibration.ir_search_scale = float(np.clip(search_scale, 1.0, 12.0))
        self.calibration.ir_sample_count = int(np.clip(sample_count, 5, 9))
        self.calibration.ir_strict_regions = bool(
            self.ir_strict_regions_var.get()
        )
        self.calibration.ir_global_fallback = bool(
            self.ir_global_fallback_var.get()
        )
        self.calibration.ir_diamond_min_score = float(
            np.clip(diamond_min_score, 0.10, 0.90)
        )
        self.calibration.ir_lock_enabled = bool(self.ir_lock_enabled_var.get())
        self.calibration.ir_lock_radius = float(
            np.clip(lock_radius, 3.0, 100.0)
        )
        self.calibration.ir_max_travel = float(
            np.clip(
                max(max_travel, self.calibration.ir_lock_radius),
                10.0,
                500.0,
            )
        )
        self.ir_settle_var.set(self.calibration.ir_settle_seconds)
        self.ir_flash_delta_var.set(self.calibration.ir_flash_delta)
        self.ir_day_black_var.set(
            self.calibration.ir_day_black_threshold
        )
        self.ir_search_scale_var.set(self.calibration.ir_search_scale)
        self.ir_diamond_min_score_var.set(
            self.calibration.ir_diamond_min_score
        )
        self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)
        self.ir_max_travel_var.set(self.calibration.ir_max_travel)

    def _on_ir_lock_change(self, event=None):
        try:
            radius = float(str(self.ir_lock_radius_var.get()).replace(",", "."))
            max_travel = float(
                str(self.ir_max_travel_var.get()).replace(",", ".")
            )
            diamond_min_score = float(
                str(self.ir_diamond_min_score_var.get()).replace(",", ".")
            )
            day_black_threshold = int(
                round(float(self.ir_day_black_var.get()))
            )
        except (tk.TclError, ValueError):
            radius = self.calibration.ir_lock_radius
            max_travel = self.calibration.ir_max_travel
            diamond_min_score = self.calibration.ir_diamond_min_score
            day_black_threshold = self.calibration.ir_day_black_threshold
        self.calibration.ir_lock_enabled = bool(self.ir_lock_enabled_var.get())
        self.calibration.ir_strict_regions = bool(
            self.ir_strict_regions_var.get()
        )
        self.calibration.ir_global_fallback = bool(
            self.ir_global_fallback_var.get()
        )
        self.calibration.ir_diamond_min_score = float(
            np.clip(diamond_min_score, 0.10, 0.90)
        )
        self.calibration.ir_day_black_threshold = int(
            np.clip(day_black_threshold, 10, 160)
        )
        self.calibration.ir_lock_radius = float(
            np.clip(radius, 3.0, 100.0)
        )
        self.calibration.ir_max_travel = float(
            np.clip(
                max(max_travel, self.calibration.ir_lock_radius),
                10.0,
                500.0,
            )
        )
        self.ir_diamond_min_score_var.set(
            self.calibration.ir_diamond_min_score
        )
        self.ir_day_black_var.set(
            self.calibration.ir_day_black_threshold
        )
        self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)
        self.ir_max_travel_var.set(self.calibration.ir_max_travel)
        if self.tracker:
            self._reset_processing_modules(announce=False)
        state = "включено" if self.calibration.ir_lock_enabled else "выключено"
        locality = (
            "отдельно по Pn"
            if self.calibration.ir_strict_regions
            else "в расширенных областях"
        )
        self.status_label.config(
            text=(
                f"Удержание ИК-метки {state}, поиск {locality}; "
                f"ход до {self.calibration.ir_max_travel:.0f} px; "
                "обработка начата заново"
            )
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
        self.calibration.show_displacements = bool(
            self.show_displacements_var.get()
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
                self.calibration.show_displacements = bool(
                    self.show_displacements_var.get()
                )
                self.calibration.close_shape = bool(self.close_shape_var.get())
            if hasattr(self, "ir_settle_var"):
                self.calibration.ir_settle_seconds = float(
                    self.ir_settle_var.get()
                )
                self.calibration.ir_flash_delta = int(
                    self.ir_flash_delta_var.get()
                )
                self.calibration.ir_day_black_threshold = int(
                    self.ir_day_black_var.get()
                )
                self.calibration.ir_search_scale = float(
                    self.ir_search_scale_var.get()
                )
                self.calibration.ir_strict_regions = bool(
                    self.ir_strict_regions_var.get()
                )
                self.calibration.ir_global_fallback = bool(
                    self.ir_global_fallback_var.get()
                )
                self.calibration.ir_diamond_min_score = float(
                    self.ir_diamond_min_score_var.get()
                )
                self.calibration.ir_lock_enabled = bool(
                    self.ir_lock_enabled_var.get()
                )
                self.calibration.ir_lock_radius = float(
                    self.ir_lock_radius_var.get()
                )
                self.calibration.ir_max_travel = float(
                    self.ir_max_travel_var.get()
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
            self.show_displacements_var.set(
                self.calibration.show_displacements
            )
            self.close_shape_var.set(self.calibration.close_shape)
        if hasattr(self, "ir_settle_var"):
            self.ir_settle_var.set(self.calibration.ir_settle_seconds)
            self.ir_flash_delta_var.set(self.calibration.ir_flash_delta)
            self.ir_day_black_var.set(
                self.calibration.ir_day_black_threshold
            )
            self.ir_search_scale_var.set(self.calibration.ir_search_scale)
            self.ir_strict_regions_var.set(
                self.calibration.ir_strict_regions
            )
            self.ir_global_fallback_var.set(
                self.calibration.ir_global_fallback
            )
            self.ir_diamond_min_score_var.set(
                self.calibration.ir_diamond_min_score
            )
            self.ir_lock_enabled_var.set(self.calibration.ir_lock_enabled)
            self.ir_lock_radius_var.set(self.calibration.ir_lock_radius)
            self.ir_max_travel_var.set(self.calibration.ir_max_travel)
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

        if name == "Базовый":
            for field, value in BASE_DISPLAY_SETTINGS.items():
                setattr(self.calibration, field, value)

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
                    "initial_x_px",
                    "initial_y_px",
                    "dx_px",
                    "dy_px",
                    "displacement_px",
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
                if track.initial_position is not None:
                    initial_x = f"{track.initial_position[0]:.4f}"
                    initial_y = f"{track.initial_position[1]:.4f}"
                    displacement_x = f"{track.displacement[0]:.4f}"
                    displacement_y = f"{track.displacement[1]:.4f}"
                    displacement = f"{track.displacement_magnitude:.4f}"
                else:
                    initial_x = initial_y = ""
                    displacement_x = displacement_y = displacement = ""
                self._record_log_writer.writerow(
                    [
                        timestamp,
                        f"{elapsed:.3f}",
                        self._record_frame_index,
                        track.id,
                        state,
                        x_value,
                        y_value,
                        initial_x,
                        initial_y,
                        displacement_x,
                        displacement_y,
                        displacement,
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
            self.status_label.config(
                text=(
                    "Отслеживание запущено; первый LOCK каждой точки "
                    "будет принят за нулевую координату"
                )
            )

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
        if self._region_selection_active:
            self._cancel_peak_region_selection()
            return
        self._update_calibration_from_ui()
        self._region_selection_active = True
        self._region_selection_frame = self.current_frame.copy()
        self._region_selection_expected = max(
            1, int(self.calibration.expected_reflectors)
        )
        self._region_selection_rects = []
        self._region_selection_start = None
        self._region_selection_current = None
        self._region_selection_previous_regions = list(
            self.calibration.peak_regions
        )
        self.main_canvas.config(cursor="crosshair")
        self.main_canvas.focus_set()
        self._region_selection_escape_bind_id = self.root.bind(
            "<Escape>", self._cancel_peak_region_selection, add="+"
        )
        self._update_region_selection_status()

    def _update_region_selection_status(self):
        next_index = len(self._region_selection_rects) + 1
        self.status_label.config(
            text=(
                f"Выбор P{next_index}/{self._region_selection_expected}: "
                "ЛКМ — рамка, колесо — масштаб, средняя кнопка — панорама, "
                "ПКМ — назад, Esc — отмена"
            )
        )

    def _leave_region_selection_mode(self):
        if self._region_selection_escape_bind_id:
            self.root.unbind(
                "<Escape>", self._region_selection_escape_bind_id
            )
        self._region_selection_escape_bind_id = None
        self._region_selection_active = False
        self._region_selection_frame = None
        self._region_selection_expected = 0
        self._region_selection_start = None
        self._region_selection_current = None
        self._region_selection_previous_regions = []
        self.main_canvas.config(
            cursor="fleur" if self.view_zoom > 1.0 else ""
        )

    def _cancel_peak_region_selection(self, event=None):
        if not self._region_selection_active:
            return
        self._region_selection_rects = []
        self._leave_region_selection_mode()
        self.status_label.config(
            text="Выбор областей отменен; прежние области сохранены"
        )
        return "break"

    def _undo_peak_region_selection(self, event=None):
        if not self._region_selection_active:
            return
        if self._region_selection_start is not None:
            self._region_selection_start = None
            self._region_selection_current = None
        elif self._region_selection_rects:
            self._region_selection_rects.pop()
        self._update_region_selection_status()
        return "break"

    def _finish_peak_region_selection(self):
        selected = list(self._region_selection_rects)
        self._region_selection_rects = []
        self._leave_region_selection_mode()
        self.calibration.peak_regions = selected
        # Ручной повторный выбор областей отменяет старую ИК-привязку.
        self.calibration.ir_reference_regions = list(selected)
        self.calibration.ir_confirmed_centers.clear()
        self.calibration.ir_confirmed_models.clear()
        self.calibration.ir_verification_active = False
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
        if self._region_selection_active:
            self._cancel_peak_region_selection()
        self.calibration.peak_regions = []
        self.calibration.roi_region = None
        self.calibration.ir_reference_regions = []
        self.calibration.ir_confirmed_centers.clear()
        self.calibration.ir_confirmed_models.clear()
        self.calibration.ir_verification_active = False
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
        if (
            self._region_selection_active
            and self._region_selection_frame is not None
        ):
            # На время разметки кадр замораживается, но сохраняются текущие
            # масштаб и панорама основного окна.
            frame = self._region_selection_frame
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
        self._view_crop_rect = (x1, y1, crop_w, crop_h)
        self._view_image_origin = (
            (canvas_w - new_w) / 2.0,
            (canvas_h - new_h) / 2.0,
        )
        self._view_frame_shape = (frame_h, frame_w)
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
        if self._region_selection_active:
            self._draw_region_selection_overlay()
        self.main_canvas.image = photo

    def _draw_region_selection_overlay(self):
        """Рисует выбранные рамки непосредственно поверх основного Canvas."""
        for index, region in enumerate(self._region_selection_previous_regions):
            x, y, width, height = region
            x1, y1 = self._frame_to_canvas(x, y)
            x2, y2 = self._frame_to_canvas(x + width, y + height)
            self.main_canvas.create_rectangle(
                x1, y1, x2, y2, outline="#777777", dash=(4, 3), width=1
            )
            self.main_canvas.create_text(
                x1 + 4,
                y1 + 4,
                text=f"стар. P{index + 1}",
                fill="#d0d0d0",
                anchor=tk.NW,
                font=("Arial", 9, "bold"),
            )

        for index, region in enumerate(self._region_selection_rects):
            x, y, width, height = region
            x1, y1 = self._frame_to_canvas(x, y)
            x2, y2 = self._frame_to_canvas(x + width, y + height)
            self.main_canvas.create_rectangle(
                x1, y1, x2, y2, outline="#00ff45", width=2
            )
            self.main_canvas.create_text(
                x1 + 4,
                y1 + 4,
                text=f"P{index + 1}",
                fill="#00ff45",
                anchor=tk.NW,
                font=("Arial", 10, "bold"),
            )

        if (
            self._region_selection_start is not None
            and self._region_selection_current is not None
        ):
            start_x, start_y = self._frame_to_canvas(
                *self._region_selection_start
            )
            end_x, end_y = self._frame_to_canvas(
                *self._region_selection_current
            )
            self.main_canvas.create_rectangle(
                start_x,
                start_y,
                end_x,
                end_y,
                outline="#00e5ff",
                dash=(5, 3),
                width=2,
            )

        current = min(
            len(self._region_selection_rects) + 1,
            max(1, self._region_selection_expected),
        )
        self.main_canvas.create_rectangle(
            8, 8, 560, 42, fill="#111111", outline="#00e5ff", width=1
        )
        self.main_canvas.create_text(
            18,
            25,
            text=(
                f"Область P{current}/{self._region_selection_expected}: "
                "ЛКМ рамка • колесо zoom • средняя кнопка панорама"
            ),
            fill="#00e5ff",
            anchor=tk.W,
            font=("Arial", 10, "bold"),
        )

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
