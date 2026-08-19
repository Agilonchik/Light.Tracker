import unittest

import cv2
import numpy as np

from day_night_reflector_detector import (
    DayNightReflectorDetector,
    DetectorSettings,
)


ROI = (30, 25, 65, 60)


def make_background() -> np.ndarray:
    yy, xx = np.indices((120, 150))
    image = (118 + 10 * np.sin(xx / 9.0) + 7 * np.cos(yy / 11.0)).astype(np.uint8)
    return image


def put_diamond(image: np.ndarray, center, radius: int, value: int) -> None:
    x, y = center
    points = np.array(
        [[x, y - radius], [x + radius, y], [x, y + radius], [x - radius, y]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, points, int(value))


def samples(image: np.ndarray, offsets=(-1, 0, 1, 0, 0)):
    result = []
    for offset in offsets:
        noisy = np.clip(image.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        result.append(noisy)
    return result


class DayNightDetectorTests(unittest.TestCase):
    def detector(self) -> DayNightReflectorDetector:
        return DayNightReflectorDetector(
            DetectorSettings(
                day_black_max=55,
                night_bright_min=175,
                min_positive_gain=100,
                min_area=4,
                max_area=600,
                minimum_score=0.35,
                registration_enabled=False,
            )
        )

    def test_repeated_black_to_bright_diamond_is_found(self):
        day_1 = make_background()
        day_2 = make_background()
        night_1 = make_background()
        night_2 = make_background()
        put_diamond(day_1, (63, 54), 7, 18)
        put_diamond(day_2, (63, 54), 7, 20)
        put_diamond(night_1, (63, 54), 8, 245)
        put_diamond(night_2, (64, 54), 8, 243)

        batch = self.detector().analyze(
            [samples(day_1), samples(day_2)],
            [samples(night_1), samples(night_2)],
            [ROI],
        )
        result = batch.results[0]
        self.assertTrue(result.found, result.reason)
        self.assertAlmostEqual(result.center[0], 63.5, delta=1.5)
        self.assertAlmostEqual(result.center[1], 54.0, delta=1.5)

    def test_darkening_of_bright_object_is_not_a_flash(self):
        day_1 = make_background()
        day_2 = make_background()
        night_1 = make_background()
        night_2 = make_background()
        put_diamond(day_1, (63, 54), 8, 245)
        put_diamond(day_2, (63, 54), 8, 243)
        put_diamond(night_1, (63, 54), 8, 18)
        put_diamond(night_2, (63, 54), 8, 20)

        batch = self.detector().analyze(
            [samples(day_1), samples(day_2)],
            [samples(night_1), samples(night_2)],
            [ROI],
        )
        self.assertFalse(batch.results[0].found)

    def test_static_bright_object_is_not_a_flash(self):
        day_1 = make_background()
        day_2 = make_background()
        night_1 = make_background()
        night_2 = make_background()
        for image in (day_1, day_2, night_1, night_2):
            put_diamond(image, (63, 54), 8, 235)

        batch = self.detector().analyze(
            [samples(day_1), samples(day_2)],
            [samples(night_1), samples(night_2)],
            [ROI],
        )
        self.assertFalse(batch.results[0].found)

    def test_candidate_outside_roi_is_never_used(self):
        day_1 = make_background()
        day_2 = make_background()
        night_1 = make_background()
        night_2 = make_background()
        for image in (day_1, day_2):
            put_diamond(image, (118, 54), 7, 15)
        for image in (night_1, night_2):
            put_diamond(image, (118, 54), 8, 250)

        batch = self.detector().analyze(
            [samples(day_1), samples(day_2)],
            [samples(night_1), samples(night_2)],
            [ROI],
        )
        self.assertFalse(batch.results[0].found)

    def test_non_repeated_flash_is_rejected(self):
        day_1 = make_background()
        day_2 = make_background()
        night_1 = make_background()
        night_2 = make_background()
        put_diamond(day_1, (50, 50), 7, 15)
        put_diamond(night_1, (50, 50), 8, 245)
        put_diamond(day_2, (82, 70), 7, 15)
        put_diamond(night_2, (82, 70), 8, 245)

        batch = self.detector().analyze(
            [samples(day_1), samples(day_2)],
            [samples(night_1), samples(night_2)],
            [ROI],
        )
        self.assertFalse(batch.results[0].found)
        self.assertEqual(batch.results[0].reason, "NOT_REPEATABLE")


if __name__ == "__main__":
    unittest.main()
