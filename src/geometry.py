import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import TrackerConfig
from tracker_types import Point, ThrusterPoints


def build_pool_mask(frame_shape: Tuple[int, int, int], corners: Optional[List[Point]]) -> Optional[np.ndarray]:
    if corners is None:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_homography(cfg: TrackerConfig) -> Optional[np.ndarray]:
    if cfg.pool_corners_px is None or cfg.pool_width_m is None or cfg.pool_height_m is None:
        return None
    src = np.array(cfg.pool_corners_px, dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(cfg.pool_width_m), 0.0],
            [float(cfg.pool_width_m), float(cfg.pool_height_m)],
            [0.0, float(cfg.pool_height_m)],
        ],
        dtype=np.float32,
    )
    homography, _status = cv2.findHomography(src, dst, method=0)
    return homography


def transform_point(homography: Optional[np.ndarray], point: Optional[Point]) -> Tuple[float, float]:
    if homography is None or point is None or not np.isfinite(point[0]) or not np.isfinite(point[1]):
        return (math.nan, math.nan)
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0, 0]
    return float(dst[0]), float(dst[1])


def dist(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def distance_stats(points: ThrusterPoints) -> Tuple[float, float]:
    if len(points) < 2:
        return (0.0, 0.0)
    pairwise = [dist(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
    return min(pairwise), max(pairwise)


def clamp_vector(vec: Point, max_norm: float) -> Point:
    norm = math.hypot(vec[0], vec[1])
    if norm <= max_norm or norm <= 1e-9:
        return vec
    scale = max_norm / norm
    return (vec[0] * scale, vec[1] * scale)
