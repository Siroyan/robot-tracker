import math
from itertools import permutations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import TrackerConfig, target_thruster_count
from detection import Detection
from geometry import clamp_vector, dist, distance_stats
from tracker_types import Point, ThrusterPoints, ThrusterVelocities


def _thruster_velocity(
    cfg: TrackerConfig,
    centroid_velocity: Point,
    point_velocity: Optional[Point],
) -> Point:
    if point_velocity is None:
        return clamp_vector(centroid_velocity, float(cfg.thruster_max_step_px))
    local_velocity = clamp_vector(point_velocity, float(cfg.thruster_max_step_px))
    centroid_velocity = clamp_vector(centroid_velocity, float(cfg.thruster_max_step_px))
    blended = (
        0.7 * local_velocity[0] + 0.3 * centroid_velocity[0],
        0.7 * local_velocity[1] + 0.3 * centroid_velocity[1],
    )
    return clamp_vector(blended, float(cfg.thruster_max_step_px))


def predict_thruster_centers(
    cfg: TrackerConfig,
    prev_thruster_points: ThrusterPoints,
    prev_thruster_velocities: Optional[ThrusterVelocities],
    centroid_velocity: Point,
) -> ThrusterPoints:
    """Predict next ROI centers from previous thruster points and velocities."""
    return [
        (
            point[0] + velocity[0],
            point[1] + velocity[1],
        )
        for i, point in enumerate(prev_thruster_points)
        for velocity in [
            _thruster_velocity(
                cfg,
                centroid_velocity,
                prev_thruster_velocities[i] if prev_thruster_velocities is not None and i < len(prev_thruster_velocities) else None,
            )
        ]
    ]


def order_thruster_points(points: ThrusterPoints, prev_points: Optional[ThrusterPoints]) -> ThrusterPoints:
    """Stabilize thruster ordering, optionally using the previous frame layout."""
    if prev_points is not None and len(points) != len(prev_points):
        return points
    if len(points) <= 1:
        return points
    if prev_points is None:
        n_points = len(points)
        centroid = (
            sum(point[0] for point in points) / n_points,
            sum(point[1] for point in points) / n_points,
        )
        return sorted(points, key=lambda point: math.atan2(point[1] - centroid[1], point[0] - centroid[0]))

    best: Optional[ThrusterPoints] = None
    best_score = math.inf
    for perm in permutations(points):
        score = sum(dist(perm[i], prev_points[i]) for i in range(len(points)))
        if score < best_score:
            best_score = score
            best = list(perm)
    return best if best is not None else points


def suppress_nearby_points(points: List[Tuple[float, float, float]], min_distance: float) -> ThrusterPoints:
    kept: ThrusterPoints = []
    for x, y, _score in sorted(points, key=lambda item: item[2], reverse=True):
        point = (float(x), float(y))
        if all(dist(point, other) >= min_distance for other in kept):
            kept.append(point)
    return kept


def dedupe_points(points: ThrusterPoints, min_distance: float) -> ThrusterPoints:
    kept: ThrusterPoints = []
    for point in points:
        if all(dist(point, other) >= min_distance for other in kept):
            kept.append(point)
    return kept


def refine_points_from_mask(mask: np.ndarray, seeds: ThrusterPoints, radius: float) -> ThrusterPoints:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    points = np.column_stack([xs, ys]).astype(np.float32)
    refined: ThrusterPoints = []
    radius2 = radius * radius
    for sx, sy in seeds:
        d2 = (points[:, 0] - sx) ** 2 + (points[:, 1] - sy) ** 2
        cluster = points[d2 <= radius2]
        if len(cluster) == 0:
            return []
        refined.append((float(cluster[:, 0].mean()), float(cluster[:, 1].mean())))
    return refined


def split_detection_points(mask: np.ndarray, detection: Detection, n_splits: int) -> ThrusterPoints:
    """Split one merged detection into multiple point candidates with k-means."""
    x0 = max(0, int(detection["x"]) - 4)
    y0 = max(0, int(detection["y"]) - 4)
    x1 = min(mask.shape[1], int(detection["x"] + detection["w"]) + 4)
    y1 = min(mask.shape[0], int(detection["y"] + detection["h"]) + 4)
    roi = mask[y0:y1, x0:x1]
    ys, xs = np.where(roi > 0)
    if len(xs) < n_splits * 4:
        return []
    points = np.column_stack([xs.astype(np.float32) + x0, ys.astype(np.float32) + y0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _compactness, _labels, centers = cv2.kmeans(points, n_splits, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    return [(float(centers[i, 0]), float(centers[i, 1])) for i in range(n_splits)]


def mask_peak_candidates(mask: np.ndarray, min_peak_score: float) -> List[Tuple[float, float, float]]:
    """Extract candidate points from the distance-transform peaks of a mask."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dt.max()) <= 0.0:
        return []
    dilated = cv2.dilate(dt, np.ones((5, 5), np.float32))
    peak_mask = np.uint8((dt >= dilated - 1e-5) & (dt >= max(min_peak_score, 0.35 * float(dt.max())))) * 255
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)
    candidates: List[Tuple[float, float, float]] = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 0:
            continue
        cx, cy = centroids[label]
        ix = int(np.clip(round(cx), 0, dt.shape[1] - 1))
        iy = int(np.clip(round(cy), 0, dt.shape[0] - 1))
        candidates.append((float(cx), float(cy), float(dt[iy, ix])))
    return candidates


def extract_candidate_points(
    mask: np.ndarray,
    detections: List[Detection],
    cfg: TrackerConfig,
    search_center: Optional[Point],
) -> ThrusterPoints:
    candidates = mask_peak_candidates(mask, 1.0)
    for detection in detections:
        score = math.sqrt(max(float(detection["area"]), 1.0))
        candidates.append((float(detection["cx"]), float(detection["cy"]), score))

    if search_center is not None:
        candidates.sort(key=lambda item: dist((item[0], item[1]), search_center))

    deduped = suppress_nearby_points(candidates, cfg.min_thruster_distance_px * 0.55)
    if search_center is not None:
        deduped.sort(key=lambda point: dist(point, search_center))
    return deduped


def select_initial_thruster_points(mask: np.ndarray, detections: List[Detection], cfg: TrackerConfig) -> ThrusterPoints:
    """Estimate initial thruster points from the first frame only."""
    n_thrusters = target_thruster_count(cfg)

    if len(detections) >= n_thrusters:
        direct = [(float(item["cx"]), float(item["cy"])) for item in detections[:n_thrusters]]
        return order_thruster_points(direct, None)
    if not detections:
        return []

    points: ThrusterPoints = [(float(item["cx"]), float(item["cy"])) for item in detections]
    remaining = n_thrusters - len(points)
    det_index = 0
    while remaining > 0 and det_index < len(detections):
        splits = min(remaining + 1, 3)
        split_points = split_detection_points(mask, detections[det_index], splits)
        if len(split_points) == splits:
            if det_index < len(points):
                points.pop(det_index)
            points.extend(split_points)
            remaining = n_thrusters - len(points)
        det_index += 1

    points = dedupe_points(points, 6.0)
    if len(points) < n_thrusters:
        points.extend(extract_candidate_points(mask, detections, cfg, cfg.init_point_px))
        points = dedupe_points(points, 6.0)
    if len(points) < n_thrusters:
        return []

    ordered = order_thruster_points(points[:n_thrusters], None)
    refined = refine_points_from_mask(mask, ordered, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
    if len(refined) == n_thrusters and distance_stats(refined)[0] >= 6.0:
        return order_thruster_points(refined, None)
    return ordered


def localize_thruster_point(mask: np.ndarray, predicted: Point, search_radius: float) -> Optional[Point]:
    """Find the best connected component near a predicted thruster position."""
    radius_px = int(math.ceil(search_radius))
    x0 = max(0, int(round(predicted[0])) - radius_px)
    x1 = min(mask.shape[1], int(round(predicted[0])) + radius_px + 1)
    y0 = max(0, int(round(predicted[1])) - radius_px)
    y1 = min(mask.shape[0], int(round(predicted[1])) + radius_px + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = mask[y0:y1, x0:x1]
    if int(np.count_nonzero(roi)) < 6:
        return None

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
    best_point: Optional[Point] = None
    best_score = math.inf
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        cx = float(centroids[label][0] + x0)
        cy = float(centroids[label][1] + y0)
        point = (cx, cy)
        jump = dist(point, predicted)
        if jump > search_radius:
            continue
        score = jump - 0.02 * area
        if score < best_score:
            best_score = score
            best_point = point
    return best_point


def _localize_thruster_point(
    mask: np.ndarray,
    relaxed_mask: np.ndarray,
    prev_point: Point,
    predicted: Point,
    search_radius: float,
    reacquire_radius: float,
) -> Optional[Point]:
    for work_mask, center, radius in (
        (mask, predicted, search_radius),
        (mask, prev_point, search_radius),
        (mask, predicted, reacquire_radius),
        (mask, prev_point, reacquire_radius),
        (relaxed_mask, predicted, search_radius),
        (relaxed_mask, prev_point, search_radius),
        (relaxed_mask, predicted, reacquire_radius),
        (relaxed_mask, prev_point, reacquire_radius),
    ):
        localized = localize_thruster_point(work_mask, center, radius)
        if localized is not None:
            return localized
    return None


def track_fixed_thrusters(
    mask: np.ndarray,
    relaxed_mask: np.ndarray,
    cfg: TrackerConfig,
    prev_thruster_points: ThrusterPoints,
    prev_thruster_velocities: Optional[ThrusterVelocities],
    centroid_velocity: Point,
) -> Tuple[ThrusterPoints, bool]:
    """Track fixed-ID thrusters within local ROIs and report whether any point was held."""
    if len(prev_thruster_points) != target_thruster_count(cfg):
        return [], False

    working_mask = mask.copy()
    tracked: ThrusterPoints = []
    had_hold = False
    search_radius = max(8.0, float(cfg.thruster_search_radius_px))
    reacquire_radius = max(search_radius, float(cfg.thruster_reacquire_radius_px))
    erase_radius = max(8, int(round(cfg.min_thruster_distance_px * 0.45)))
    predicted_points = predict_thruster_centers(cfg, prev_thruster_points, prev_thruster_velocities, centroid_velocity)
    for prev_point, predicted in zip(prev_thruster_points, predicted_points):
        localized = _localize_thruster_point(working_mask, relaxed_mask, prev_point, predicted, search_radius, reacquire_radius)
        used_hold = localized is None
        localized = prev_point if localized is None else localized
        had_hold |= used_hold
        if any(dist(localized, other) < cfg.min_thruster_distance_px * 0.6 for other in tracked):
            return [], False
        tracked.append(localized)
        if not used_hold:
            cv2.circle(working_mask, (int(round(localized[0])), int(round(localized[1]))), erase_radius, 0, thickness=-1)

    min_pair_distance, max_pair_distance = distance_stats(tracked)
    if min_pair_distance < cfg.min_thruster_distance_px * 0.55:
        return [], False
    if max_pair_distance > cfg.max_thruster_distance_px * 1.25:
        return [], False
    return tracked, had_hold


def estimate_thruster_points(
    mask: np.ndarray,
    detections: List[Detection],
    cfg: TrackerConfig,
    prev_pos: Optional[Point],
    velocity: Point,
    prev_thruster_points: Optional[ThrusterPoints] = None,
) -> ThrusterPoints:
    """Estimate all thruster points from the current frame without fixed-ID ROI tracking."""
    n_thrusters = target_thruster_count(cfg)

    def is_valid(points: ThrusterPoints) -> bool:
        if len(points) != n_thrusters:
            return False
        min_pair_distance, max_pair_distance = distance_stats(points)
        return min_pair_distance >= cfg.min_thruster_distance_px and max_pair_distance <= cfg.max_thruster_distance_px

    ys, xs = np.where(mask > 0)
    if len(xs) < n_thrusters:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    search_center: Optional[Point] = None
    if prev_pos is not None:
        search_center = (prev_pos[0] + velocity[0], prev_pos[1] + velocity[1])
    elif cfg.init_point_px is not None:
        search_center = cfg.init_point_px

    if search_center is not None:
        search_radius = max(cfg.max_jump_px * 2.0, cfg.cluster_radius_px * 4.5)
        d2 = (points[:, 0] - float(search_center[0])) ** 2 + (points[:, 1] - float(search_center[1])) ** 2
        filtered = points[d2 <= search_radius * search_radius]
        if len(filtered) >= n_thrusters:
            points = filtered

    if len(points) > 4000:
        idx = np.linspace(0, len(points) - 1, 4000, dtype=np.int32)
        points = points[idx]

    work_mask = mask.copy()
    if search_center is not None:
        roi_radius = max(cfg.max_jump_px * 1.25, cfg.cluster_radius_px * 3.2)
        roi = np.zeros_like(mask)
        cv2.circle(roi, (int(search_center[0]), int(search_center[1])), int(roi_radius), 255, -1)
        cropped = cv2.bitwise_and(mask, roi)
        if int(cropped.sum() // 255) >= 12:
            work_mask = cropped

    peak_candidates = mask_peak_candidates(work_mask, 1.5)
    if peak_candidates:
        seeds = suppress_nearby_points(peak_candidates, cfg.min_thruster_distance_px)
        if len(seeds) == n_thrusters:
            refined = refine_points_from_mask(work_mask, seeds, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
            refined = order_thruster_points(refined, prev_thruster_points)
            if is_valid(refined):
                return refined

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _compactness, _labels, centers = cv2.kmeans(points, n_thrusters, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    thruster_points = [(float(centers[i, 0]), float(centers[i, 1])) for i in range(n_thrusters)]
    thruster_points = refine_points_from_mask(work_mask, thruster_points, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
    thruster_points = order_thruster_points(thruster_points, prev_thruster_points)
    return thruster_points if len(thruster_points) == n_thrusters else []


def choose_robot_position(
    detections: List[Detection],
    mask: np.ndarray,
    relaxed_mask: np.ndarray,
    cfg: TrackerConfig,
    prev_pos: Optional[Point],
    velocity: Point,
    prev_thruster_points: Optional[ThrusterPoints],
    prev_thruster_velocities: Optional[ThrusterVelocities],
) -> Tuple[ThrusterPoints, Optional[Point], float, int, bool, str]:
    """Choose thruster points and robot centroid for one frame.

    Returns tracked thruster points, centroid, orange area, detected point count,
    reliability flag, and tracking mode (`init`, `roi`, `hold`, `global`, `none`).
    """
    if not detections:
        return [], None, 0.0, 0, False, "none"

    n_thrusters = target_thruster_count(cfg)
    thruster_points, tracking_mode = [], "global"
    if prev_thruster_points is not None and len(prev_thruster_points) == n_thrusters:
        thruster_points, had_hold = track_fixed_thrusters(mask, relaxed_mask, cfg, prev_thruster_points, prev_thruster_velocities, velocity)
        tracking_mode = "hold" if had_hold else "roi"
    if len(thruster_points) != n_thrusters:
        thruster_points, tracking_mode = estimate_thruster_points(mask, detections, cfg, prev_pos, velocity, prev_thruster_points), "global"
    if len(thruster_points) != n_thrusters:
        return [], None, 0.0, len(thruster_points), False, "none"

    centroid = tuple(float(sum(point[i] for point in thruster_points) / n_thrusters) for i in (0, 1))
    total_area = float(sum(max(detection["area"], 1.0) for detection in detections))

    reliable = True
    if prev_pos is not None:
        predicted = (prev_pos[0] + velocity[0], prev_pos[1] + velocity[1])
        reacquire_limit = max(cfg.max_jump_px * 2.0, cfg.cluster_radius_px * 3.0)
        reliable = dist(centroid, predicted) <= cfg.max_jump_px or dist(centroid, prev_pos) <= reacquire_limit
        if not reliable:
            return [], None, total_area, len(thruster_points), False, tracking_mode
    return thruster_points, centroid, total_area, len(thruster_points), reliable, tracking_mode
