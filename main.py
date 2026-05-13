#!/usr/bin/env python3
"""
Underwater robot tracker using orange thruster color + perspective transform.

Outputs frame-by-frame robot position as CSV. If pool corners and real pool size are
provided, pixel coordinates are converted to metric pool coordinates using homography.

Typical workflow:
  1) Export a reference frame and read pixel coordinates in any image viewer:
     python underwater_robot_tracker_v3_headless.py input.mp4 \
       --reference-frame 0 --export-reference-frame reference.jpg

  2) Create a config without GUI:
     python underwater_robot_tracker_v3_headless.py input.mp4 --make-config config.json \
       --pool-width-m 2.0 --pool-height-m 3.0 \
       --pool-corners-px 100,120 900,110 930,690 80,700 \
       --init-point-px 420,360

  3) Track:
     python underwater_robot_tracker_v3_headless.py input.mp4 --config config.json \
       --csv positions.csv --annotated annotated.mp4

Corner order is: top-left, top-right, bottom-right, bottom-left in the video image.
Destination metric coordinate is:
  top-left=(0,0), top-right=(pool_width_m,0),
  bottom-right=(pool_width_m,pool_height_m), bottom-left=(0,pool_height_m)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import csv
from dataclasses import dataclass, asdict
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
ThrusterPoints = List[Point]


@dataclass
class TrackerConfig:
    # HSV threshold for orange objects. OpenCV hue range is 0..179.
    hsv_lower: Tuple[int, int, int] = (0, 40, 20)
    hsv_upper: Tuple[int, int, int] = (40, 255, 255)

    # Contour filters.
    min_area_px: float = 5.0
    max_area_px: float = 30000.0

    # Tracking parameters.
    # If the robot has multiple orange thrusters, contours inside cluster_radius_px
    # around the chosen contour are averaged as one robot position.
    cluster_radius_px: float = 80.0
    max_jump_px: float = 180.0
    smoothing_alpha: float = 0.35  # 0=no update, 1=no smoothing
    min_thruster_distance_px: float = 50.0
    max_thruster_distance_px: float = 190.0
    orange_clahe_clip_limit: float = 2.0
    orange_red_minus_green_min: int = 8
    orange_green_minus_blue_min: int = -5
    orange_min_red: int = 35
    orange_min_green: int = 20

    # Optional image-space polygon to suppress orange objects outside the pool.
    # Order: top-left, top-right, bottom-right, bottom-left.
    pool_corners_px: Optional[List[Point]] = None

    # Real pool size for homography.
    pool_width_m: Optional[float] = None
    pool_height_m: Optional[float] = None

    # Optional initial robot position in pixels. Strongly recommended if there are
    # other orange objects in the image.
    init_point_px: Optional[Point] = None

    # Optional frame index used for manual config creation.
    reference_frame: int = 0


def load_config(path: Optional[str]) -> TrackerConfig:
    cfg = TrackerConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # normalize list types
    cfg.hsv_lower = tuple(cfg.hsv_lower)  # type: ignore[arg-type]
    cfg.hsv_upper = tuple(cfg.hsv_upper)  # type: ignore[arg-type]
    if cfg.pool_corners_px is not None:
        cfg.pool_corners_px = [(float(x), float(y)) for x, y in cfg.pool_corners_px]
    if cfg.init_point_px is not None:
        cfg.init_point_px = (float(cfg.init_point_px[0]), float(cfg.init_point_px[1]))
    return cfg


def save_config(cfg: TrackerConfig, path: str) -> None:
    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def parse_point_text(text: str) -> Point:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Point must be x,y format: {text}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Point must be numeric x,y format: {text}") from exc


def save_reference_frame(video_path: str, frame_index: int, output_path: str) -> None:
    frame = read_frame(video_path, frame_index)
    ok = cv2.imwrite(output_path, frame)
    if not ok:
        raise RuntimeError(f"Cannot write reference frame: {output_path}")
    print(f"Saved reference frame: {output_path}")


def save_orange_preview(video_path: str, frame_index: int, output_path: str, cfg: TrackerConfig) -> None:
    frame = read_frame(video_path, frame_index)
    pool_mask = build_pool_mask(frame.shape, cfg.pool_corners_px)
    detections, mask = detect_orange_contours(frame, cfg, pool_mask)
    out = frame.copy()
    for i, d in enumerate(detections[:50], start=1):
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        cx, cy = int(d["cx"]), int(d["cy"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 128, 255), 2)
        cv2.circle(out, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(out, str(i), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, str(i), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Cannot write orange preview: {output_path}")
    print(f"Saved orange preview: {output_path}")
    print("Orange candidates on reference frame:")
    for i, d in enumerate(detections[:50], start=1):
        print(f"  #{i}: cx={d['cx']:.1f}, cy={d['cy']:.1f}, area={d['area']:.1f}, bbox=({d['x']:.0f},{d['y']:.0f},{d['w']:.0f},{d['h']:.0f})")


def make_config_headless(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.reference_frame = args.reference_frame
    cfg.pool_width_m = args.pool_width_m
    cfg.pool_height_m = args.pool_height_m

    if args.pool_corners_px is not None:
        if len(args.pool_corners_px) != 4:
            raise RuntimeError("--pool-corners-px requires exactly 4 points: TL TR BR BL")
        cfg.pool_corners_px = list(args.pool_corners_px)
    elif cfg.pool_width_m is not None or cfg.pool_height_m is not None:
        print("Warning: pool size is set but --pool-corners-px was not provided. Metric conversion will be disabled.")

    if args.init_point_px is not None:
        cfg.init_point_px = args.init_point_px
    elif cfg.init_point_px is None:
        print("Warning: --init-point-px was not provided. The tracker will start from the largest orange object, which may be wrong.")

    save_config(cfg, args.make_config)
    print(f"Saved config: {args.make_config}")


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
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def transform_point(H: Optional[np.ndarray], p: Optional[Point]) -> Tuple[float, float]:
    if H is None or p is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
        return (math.nan, math.nan)
    src = np.array([[[p[0], p[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)[0, 0]
    return float(dst[0]), float(dst[1])


def enhance_frame_for_orange(frame: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.orange_clahe_clip_limit), tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def build_color_mask(
    frame: np.ndarray,
    cfg: TrackerConfig,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
    pool_mask: Optional[np.ndarray],
    *,
    open_kernel_size: int,
    open_iterations: int,
    close_kernel_size: int,
    close_iterations: int,
) -> np.ndarray:
    enhanced = enhance_frame_for_orange(frame, cfg)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower, upper)
    bgr_b, bgr_g, bgr_r = cv2.split(enhanced)
    orange_dominance = (
        (bgr_r.astype(np.int16) >= bgr_g.astype(np.int16) + int(cfg.orange_red_minus_green_min))
        & (bgr_g.astype(np.int16) >= bgr_b.astype(np.int16) + int(cfg.orange_green_minus_blue_min))
        & (bgr_r >= int(cfg.orange_min_red))
        & (bgr_g >= int(cfg.orange_min_green))
    )
    mask = cv2.bitwise_and(mask_hsv, orange_dominance.astype(np.uint8) * 255)
    if pool_mask is not None:
        mask = cv2.bitwise_and(mask, pool_mask)

    if open_iterations > 0:
        kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    if close_iterations > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = cv2.medianBlur(mask, 3)
    return mask


def extract_detections_from_mask(mask: np.ndarray, cfg: TrackerConfig) -> List[Dict[str, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Dict[str, float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if abs(M["m00"]) > 1e-9:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        detections.append(
            {
                "cx": cx,
                "cy": cy,
                "area": area,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            }
        )
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


def detect_orange_contours(frame: np.ndarray, cfg: TrackerConfig, pool_mask: Optional[np.ndarray]) -> Tuple[List[Dict[str, float]], np.ndarray]:
    mask = build_color_mask(
        frame,
        cfg,
        cfg.hsv_lower,
        cfg.hsv_upper,
        pool_mask,
        open_kernel_size=3,
        open_iterations=0,
        close_kernel_size=3,
        close_iterations=1,
    )
    detections = extract_detections_from_mask(mask, cfg)
    if detections:
        return detections, mask

    # Fallback: widen HSV range and avoid aggressive erosion when the thrusters are
    # dim or only occupy a few pixels in the frame.
    relaxed_lower = (
        max(0, int(cfg.hsv_lower[0]) - 10),
        max(0, int(cfg.hsv_lower[1]) - 40),
        max(0, int(cfg.hsv_lower[2]) - 30),
    )
    relaxed_upper = (
        min(179, int(cfg.hsv_upper[0]) + 20),
        min(255, int(cfg.hsv_upper[1])),
        min(255, int(cfg.hsv_upper[2])),
    )
    fallback_mask = build_color_mask(
        frame,
        cfg,
        relaxed_lower,
        relaxed_upper,
        pool_mask,
        open_kernel_size=3,
        open_iterations=0,
        close_kernel_size=3,
        close_iterations=0,
    )
    fallback_detections = extract_detections_from_mask(fallback_mask, cfg)
    return fallback_detections, fallback_mask


def dist(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def order_thruster_points(points: ThrusterPoints, prev_points: Optional[ThrusterPoints]) -> ThrusterPoints:
    if len(points) != 4:
        return points
    if prev_points is None or len(prev_points) != 4:
        centroid = (
            sum(p[0] for p in points) / 4.0,
            sum(p[1] for p in points) / 4.0,
        )
        return sorted(points, key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))

    best: Optional[ThrusterPoints] = None
    best_score = math.inf
    for perm in permutations(points):
        score = sum(dist(perm[i], prev_points[i]) for i in range(4))
        if score < best_score:
            best_score = score
            best = list(perm)
    return best if best is not None else points


def distance_stats(points: ThrusterPoints) -> Tuple[float, float]:
    pairwise = [dist(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
    return min(pairwise), max(pairwise)


def suppress_nearby_points(points: List[Tuple[float, float, float]], min_distance: float) -> ThrusterPoints:
    kept: ThrusterPoints = []
    for x, y, _score in sorted(points, key=lambda t: t[2], reverse=True):
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


def extract_candidate_points(
    mask: np.ndarray,
    detections: List[Dict[str, float]],
    cfg: TrackerConfig,
    search_center: Optional[Point],
) -> ThrusterPoints:
    candidates: List[Tuple[float, float, float]] = []

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dt.max()) > 0.0:
        dilated = cv2.dilate(dt, np.ones((5, 5), np.float32))
        peak_mask = np.uint8((dt >= dilated - 1e-5) & (dt >= max(1.0, 0.35 * float(dt.max())))) * 255
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] <= 0:
                continue
            cx, cy = centroids[label]
            ix = int(np.clip(round(cx), 0, dt.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, dt.shape[0] - 1))
            score = float(dt[iy, ix])
            candidates.append((float(cx), float(cy), score))

    for det in detections:
        score = math.sqrt(max(float(det["area"]), 1.0))
        candidates.append((float(det["cx"]), float(det["cy"]), score))

    if search_center is not None:
        candidates.sort(key=lambda t: dist((t[0], t[1]), search_center))

    deduped = suppress_nearby_points(candidates, cfg.min_thruster_distance_px * 0.55)
    if search_center is not None:
        deduped.sort(key=lambda p: dist(p, search_center))
    return deduped


def build_support_mask_from_top_detections(mask: np.ndarray, detections: List[Dict[str, float]]) -> np.ndarray:
    if not detections:
        return mask
    support = np.zeros_like(mask)
    for det in detections[:3]:
        x = max(0, int(det["x"]) - 10)
        y = max(0, int(det["y"]) - 10)
        x2 = min(mask.shape[1], int(det["x"] + det["w"]) + 10)
        y2 = min(mask.shape[0], int(det["y"] + det["h"]) + 10)
        support[y:y2, x:x2] = 255
    cropped = cv2.bitwise_and(mask, support)
    return cropped if int(cropped.sum() // 255) >= 16 else mask


def select_initial_thruster_points(
    mask: np.ndarray,
    detections: List[Dict[str, float]],
    cfg: TrackerConfig,
) -> ThrusterPoints:
    def split_detection(det: Dict[str, float], n_splits: int) -> ThrusterPoints:
        x0 = max(0, int(det["x"]) - 4)
        y0 = max(0, int(det["y"]) - 4)
        x1 = min(mask.shape[1], int(det["x"] + det["w"]) + 4)
        y1 = min(mask.shape[0], int(det["y"] + det["h"]) + 4)
        roi = mask[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) < n_splits * 4:
            return []
        pts = np.column_stack([xs.astype(np.float32) + x0, ys.astype(np.float32) + y0])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _compactness, _labels, centers = cv2.kmeans(pts, n_splits, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
        return [(float(centers[i, 0]), float(centers[i, 1])) for i in range(n_splits)]

    candidates = extract_candidate_points(mask, detections, cfg, cfg.init_point_px)
    for det in detections[:4]:
        candidates.append((float(det["cx"]), float(det["cy"])))
    if detections:
        candidates.extend(split_detection(detections[0], 2))
    if len(detections) >= 2:
        candidates.extend(split_detection(detections[1], 2))
    if len(detections) == 3:
        candidates.extend(split_detection(detections[0], 3))

    candidates = dedupe_points(candidates, 6.0)
    if cfg.init_point_px is not None:
        candidates.sort(key=lambda p: dist(p, cfg.init_point_px))
    if len(candidates) < 4:
        return []

    best: ThrusterPoints = []
    best_score = math.inf
    for combo in combinations(candidates[:14], 4):
        points = list(combo)
        min_pair_distance, max_pair_distance = distance_stats(points)
        if min_pair_distance < 8.0:
            continue
        if max_pair_distance > cfg.max_thruster_distance_px * 2.2:
            continue

        centroid = (
            sum(p[0] for p in points) / 4.0,
            sum(p[1] for p in points) / 4.0,
        )
        ordered = order_thruster_points(points, None)
        angles = [math.atan2(p[1] - centroid[1], p[0] - centroid[0]) for p in ordered]
        unwrapped = angles + [angles[0] + 2.0 * math.pi]
        gap_penalty = sum(abs((unwrapped[i + 1] - unwrapped[i]) - (math.pi / 2.0)) for i in range(4))
        score = 1.2 * gap_penalty + 0.02 * max_pair_distance - 0.01 * min_pair_distance
        if cfg.init_point_px is not None:
            score += 0.25 * dist(centroid, cfg.init_point_px)
        score += sum(min(dist(p, q) for q in candidates[:14] if q != p) for p in points) * 0.03
        if score < best_score:
            best_score = score
            best = points

    if len(best) == 4:
        ordered_best = order_thruster_points(best, None)
        refined = refine_points_from_mask(mask, best, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
        refined = order_thruster_points(refined, None)
        if len(refined) == 4 and distance_stats(refined)[0] >= 6.0:
            return refined
        return ordered_best
    return []


def localize_thruster_point(
    mask: np.ndarray,
    predicted: Point,
    cfg: TrackerConfig,
    search_radius: float,
) -> Optional[Point]:
    radius_px = int(math.ceil(search_radius))
    x0 = max(0, int(round(predicted[0])) - radius_px)
    x1 = min(mask.shape[1], int(round(predicted[0])) + radius_px + 1)
    y0 = max(0, int(round(predicted[1])) - radius_px)
    y1 = min(mask.shape[0], int(round(predicted[1])) + radius_px + 1)
    if x1 > x0 and y1 > y0:
        roi = mask[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) >= 8:
            global_x = xs.astype(np.float32) + x0
            global_y = ys.astype(np.float32) + y0
            d2 = (global_x - predicted[0]) ** 2 + (global_y - predicted[1]) ** 2
            keep = d2 <= search_radius * search_radius
            if int(np.count_nonzero(keep)) >= 8:
                return (float(global_x[keep].mean()), float(global_y[keep].mean()))
    return None


def track_fixed_thrusters(
    mask: np.ndarray,
    cfg: TrackerConfig,
    prev_thruster_points: ThrusterPoints,
    velocity: Point,
) -> ThrusterPoints:
    if len(prev_thruster_points) != 4:
        return []

    tracked: ThrusterPoints = []
    search_radius = max(cfg.min_thruster_distance_px * 0.9, 24.0)
    for prev_point in prev_thruster_points:
        predicted = (prev_point[0] + velocity[0], prev_point[1] + velocity[1])
        localized = localize_thruster_point(mask, predicted, cfg, search_radius)
        if localized is None:
            localized = localize_thruster_point(mask, predicted, cfg, search_radius * 1.6)
        if localized is None:
            return []
        if any(dist(localized, other) < cfg.min_thruster_distance_px * 0.6 for other in tracked):
            return []
        tracked.append(localized)

    min_pair_distance, max_pair_distance = distance_stats(tracked)
    if min_pair_distance < cfg.min_thruster_distance_px * 0.55:
        return []
    if max_pair_distance > cfg.max_thruster_distance_px * 1.25:
        return []
    return tracked


def estimate_thruster_points(
    mask: np.ndarray,
    detections: List[Dict[str, float]],
    cfg: TrackerConfig,
    prev_pos: Optional[Point],
    velocity: Point,
    prev_thruster_points: Optional[ThrusterPoints] = None,
) -> ThrusterPoints:
    def is_valid(points: ThrusterPoints) -> bool:
        if len(points) != 4:
            return False
        min_pair_distance, max_pair_distance = distance_stats(points)
        return (
            min_pair_distance >= cfg.min_thruster_distance_px
            and max_pair_distance <= cfg.max_thruster_distance_px
        )

    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
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
        if len(filtered) >= 4:
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

    dt = cv2.distanceTransform(work_mask, cv2.DIST_L2, 5)
    if float(dt.max()) > 0.0:
        dilated = cv2.dilate(dt, np.ones((5, 5), np.float32))
        peak_mask = np.uint8((dt >= dilated - 1e-5) & (dt >= max(1.5, 0.45 * float(dt.max())))) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)
        peak_candidates: List[Tuple[float, float, float]] = []
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] <= 0:
                continue
            cx, cy = centroids[label]
            ix = int(np.clip(round(cx), 0, dt.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, dt.shape[0] - 1))
            score = float(dt[iy, ix])
            peak_candidates.append((float(cx), float(cy), score))
        seeds = suppress_nearby_points(peak_candidates, cfg.min_thruster_distance_px)
        if len(seeds) == 4:
            refined = refine_points_from_mask(work_mask, seeds, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
            refined = order_thruster_points(refined, prev_thruster_points)
            if is_valid(refined):
                return refined

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _compactness, _labels, centers = cv2.kmeans(points, 4, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    thruster_points = [(float(centers[i, 0]), float(centers[i, 1])) for i in range(4)]
    thruster_points = refine_points_from_mask(work_mask, thruster_points, radius=max(cfg.min_thruster_distance_px * 0.75, 20.0))
    thruster_points = order_thruster_points(thruster_points, prev_thruster_points)
    return thruster_points if len(thruster_points) == 4 else []


def choose_robot_position(
    detections: List[Dict[str, float]],
    mask: np.ndarray,
    cfg: TrackerConfig,
    prev_pos: Optional[Point],
    velocity: Point,
    prev_thruster_points: Optional[ThrusterPoints],
) -> Tuple[ThrusterPoints, Optional[Point], float, int, bool]:
    """
    Estimates four thruster points from the orange mask and returns their centroid.
    Returns: thruster_points, centroid_px, total_area, n_thrusters, detected_reliably
    """
    if not detections:
        return [], None, 0.0, 0, False

    thruster_points = estimate_thruster_points(mask, detections, cfg, prev_pos, velocity, prev_thruster_points)
    if len(thruster_points) != 4:
        return [], None, 0.0, len(thruster_points), False

    cx = sum(p[0] for p in thruster_points) / 4.0
    cy = sum(p[1] for p in thruster_points) / 4.0
    centroid = (float(cx), float(cy))
    total_area = float(sum(max(d["area"], 1.0) for d in detections))

    if prev_pos is None:
        reliable = True
    else:
        predicted = (prev_pos[0] + velocity[0], prev_pos[1] + velocity[1])
        jump = dist(centroid, predicted)
        reliable = jump <= cfg.max_jump_px
        if not reliable:
            reacquire_limit = max(cfg.max_jump_px * 2.0, cfg.cluster_radius_px * 3.0)
            jump = dist(centroid, prev_pos)
            reliable = jump <= reacquire_limit
            if not reliable:
                return [], None, total_area, len(thruster_points), False
    return thruster_points, centroid, total_area, len(thruster_points), reliable


def compute_speeds(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add speed_mps to rows without requiring pandas."""
    last_row: Optional[Dict[str, Any]] = None
    for row in rows:
        row["speed_mps"] = math.nan
        if not bool(row["detected"]):
            continue
        if not np.isfinite(row["pool_x_m"]) or not np.isfinite(row["pool_y_m"]):
            continue
        if last_row is not None:
            dt = float(row["time_s"] - last_row["time_s"])
            if dt > 0:
                dx = float(row["pool_x_m"] - last_row["pool_x_m"])
                dy = float(row["pool_y_m"] - last_row["pool_y_m"])
                row["speed_mps"] = math.hypot(dx, dy) / dt
        last_row = row
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "frame",
        "time_s",
        "detected",
        "px_x",
        "px_y",
        "pool_x_m",
        "pool_y_m",
        "speed_mps",
        "orange_area_px2",
        "cluster_contours",
        "num_orange_candidates",
        "thruster_1_x",
        "thruster_1_y",
        "thruster_2_x",
        "thruster_2_y",
        "thruster_3_x",
        "thruster_3_y",
        "thruster_4_x",
        "thruster_4_y",
        "thruster_min_distance_px",
        "thruster_max_distance_px",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, math.nan) for k in fieldnames})


def draw_annotation(
    frame: np.ndarray,
    detections: List[Dict[str, float]],
    thruster_points: ThrusterPoints,
    pos_px: Optional[Point],
    pool_xy: Tuple[float, float],
    trail: List[Point],
    cfg: TrackerConfig,
    frame_idx: int,
    time_s: float,
    detected: bool,
) -> np.ndarray:
    out = frame.copy()

    if cfg.pool_corners_px is not None:
        pts = np.array(cfg.pool_corners_px, dtype=np.int32)
        cv2.polylines(out, [pts], True, (255, 255, 0), 2)

    for d in detections[:20]:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 128, 255), 1)

    if len(trail) >= 2:
        pts = np.array([(int(x), int(y)) for x, y in trail[-200:]], dtype=np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 0), 2)

    for i, point in enumerate(thruster_points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(out, (x, y), 7, (0, 165, 255), 2)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)

    if pos_px is not None:
        x, y = int(pos_px[0]), int(pos_px[1])
        cv2.circle(out, (x, y), 12, (0, 255, 0) if detected else (0, 0, 255), 2)
        cv2.drawMarker(out, (x, y), (0, 255, 0) if detected else (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    label1 = f"frame={frame_idx}  t={time_s:.3f}s  detected={detected}"
    if np.isfinite(pool_xy[0]) and np.isfinite(pool_xy[1]):
        label2 = f"x={pool_xy[0]:.3f} m, y={pool_xy[1]:.3f} m"
    elif pos_px is not None:
        label2 = f"px=({pos_px[0]:.1f}, {pos_px[1]:.1f})"
    else:
        label2 = "position=NaN"
    label3 = f"thrusters={len(thruster_points)}/4"
    for i, text in enumerate([label1, label2, label3]):
        y = 35 + i * 32
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def track_video(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = load_config(args.config)
    if args.pool_width_m is not None:
        cfg.pool_width_m = args.pool_width_m
    if args.pool_height_m is not None:
        cfg.pool_height_m = args.pool_height_m

    # CLI HSV override
    if args.hsv_lower is not None:
        cfg.hsv_lower = tuple(args.hsv_lower)  # type: ignore[assignment]
    if args.hsv_upper is not None:
        cfg.hsv_upper = tuple(args.hsv_upper)  # type: ignore[assignment]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first_ok, first_frame = cap.read()
    if not first_ok:
        raise RuntimeError("Cannot read first frame")
    pool_mask = build_pool_mask(first_frame.shape, cfg.pool_corners_px)
    H = build_homography(cfg)
    initial_detections, initial_mask = detect_orange_contours(first_frame, cfg, pool_mask)
    initial_thruster_points = select_initial_thruster_points(initial_mask, initial_detections, cfg)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    writer = None
    if args.annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotated, fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write annotated video: {args.annotated}")

    rows: List[Dict[str, Any]] = []
    prev_pos: Optional[Point] = None
    prev_thruster_points: Optional[ThrusterPoints] = initial_thruster_points if len(initial_thruster_points) == 4 else None
    smoothed_pos: Optional[Point] = None
    velocity: Point = (0.0, 0.0)
    trail: List[Point] = []

    frame_idx = 0
    pending_first_frame: Optional[np.ndarray] = first_frame
    pending_first_detections = initial_detections
    pending_first_mask = initial_mask
    while True:
        if pending_first_frame is not None:
            frame = pending_first_frame
            detections = pending_first_detections
            mask = pending_first_mask
            pending_first_frame = None
        else:
            ok, frame = cap.read()
            if not ok:
                break
            detections, mask = detect_orange_contours(frame, cfg, pool_mask)
        time_s = frame_idx / fps if fps > 0 else float(frame_idx)
        if frame_idx == 0 and len(initial_thruster_points) == 4:
            thruster_points = initial_thruster_points
            raw_pos = (
                sum(p[0] for p in thruster_points) / 4.0,
                sum(p[1] for p in thruster_points) / 4.0,
            )
            area = float(sum(max(d["area"], 1.0) for d in detections))
            n_cluster = 4
            reliable = True
        else:
            thruster_points, raw_pos, area, n_cluster, reliable = choose_robot_position(
                detections,
                mask,
                cfg,
                prev_pos,
                velocity,
                prev_thruster_points,
            )
        detected = raw_pos is not None and reliable

        if detected and raw_pos is not None:
            if smoothed_pos is None:
                smoothed_pos = raw_pos
            else:
                a = float(cfg.smoothing_alpha)
                smoothed_pos = (
                    (1.0 - a) * smoothed_pos[0] + a * raw_pos[0],
                    (1.0 - a) * smoothed_pos[1] + a * raw_pos[1],
                )
            if prev_pos is not None:
                velocity = (smoothed_pos[0] - prev_pos[0], smoothed_pos[1] - prev_pos[1])
            prev_pos = smoothed_pos
            prev_thruster_points = thruster_points
            pos_for_output: Optional[Point] = smoothed_pos
            trail.append(smoothed_pos)
        else:
            pos_for_output = None

        pool_xy = transform_point(H, pos_for_output)
        thruster_min_distance = math.nan
        thruster_max_distance = math.nan
        thruster_xy = [(math.nan, math.nan)] * 4
        if len(thruster_points) == 4:
            thruster_xy = [(p[0], p[1]) for p in thruster_points]
            thruster_min_distance, thruster_max_distance = distance_stats(thruster_points)
        rows.append(
            {
                "frame": frame_idx,
                "time_s": time_s,
                "detected": bool(detected),
                "px_x": pos_for_output[0] if pos_for_output is not None else math.nan,
                "px_y": pos_for_output[1] if pos_for_output is not None else math.nan,
                "pool_x_m": pool_xy[0],
                "pool_y_m": pool_xy[1],
                "orange_area_px2": area if detected else 0.0,
                "cluster_contours": n_cluster if detected else 0,
                "num_orange_candidates": len(detections),
                "thruster_1_x": thruster_xy[0][0],
                "thruster_1_y": thruster_xy[0][1],
                "thruster_2_x": thruster_xy[1][0],
                "thruster_2_y": thruster_xy[1][1],
                "thruster_3_x": thruster_xy[2][0],
                "thruster_3_y": thruster_xy[2][1],
                "thruster_4_x": thruster_xy[3][0],
                "thruster_4_y": thruster_xy[3][1],
                "thruster_min_distance_px": thruster_min_distance,
                "thruster_max_distance_px": thruster_max_distance,
            }
        )

        if writer is not None:
            annotated = draw_annotation(frame, detections, thruster_points, pos_for_output, pool_xy, trail, cfg, frame_idx, time_s, detected)
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if H is not None:
        rows = compute_speeds(rows)
    else:
        for row in rows:
            row["speed_mps"] = math.nan

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Saved CSV: {args.csv}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless tracker for an underwater robot from orange thrusters and pool homography.")
    p.add_argument("video", help="Input video path")
    p.add_argument("--config", help="JSON config path")
    p.add_argument("--make-config", help="Create JSON config and exit. This headless version does not open GUI windows.")
    p.add_argument("--reference-frame", type=int, default=0, help="Frame index used for reference/config creation")
    p.add_argument("--export-reference-frame", help="Save the reference frame as an image and exit unless --make-config is also specified")
    p.add_argument("--export-orange-preview", help="Save a reference image with detected orange candidates and exit unless --make-config is also specified")
    p.add_argument("--pool-corners-px", nargs=4, type=parse_point_text, metavar=("TL", "TR", "BR", "BL"), help="Pool corners in pixels: x,y x,y x,y x,y")
    p.add_argument("--init-point-px", type=parse_point_text, help="Initial robot/thruster point in pixels: x,y")
    p.add_argument("--pool-width-m", type=float, help="Real pool width in meters, mapped from top-left to top-right")
    p.add_argument("--pool-height-m", type=float, help="Real pool height/length in meters, mapped from top-left to bottom-left")
    p.add_argument("--csv", default="positions.csv", help="Output CSV path")
    p.add_argument("--annotated", help="Optional annotated MP4 output path")
    p.add_argument("--hsv-lower", nargs=3, type=int, metavar=("H", "S", "V"), help="Override HSV lower threshold")
    p.add_argument("--hsv-upper", nargs=3, type=int, metavar=("H", "S", "V"), help="Override HSV upper threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_for_preview = load_config(args.config)
    if args.hsv_lower is not None:
        cfg_for_preview.hsv_lower = tuple(args.hsv_lower)
    if args.hsv_upper is not None:
        cfg_for_preview.hsv_upper = tuple(args.hsv_upper)
    if args.pool_corners_px is not None:
        cfg_for_preview.pool_corners_px = list(args.pool_corners_px)

    did_export = False
    if args.export_reference_frame:
        save_reference_frame(args.video, args.reference_frame, args.export_reference_frame)
        did_export = True
    if args.export_orange_preview:
        save_orange_preview(args.video, args.reference_frame, args.export_orange_preview, cfg_for_preview)
        did_export = True

    if args.make_config:
        make_config_headless(args)
        return
    if did_export:
        return

    rows = track_video(args)
    detected_rate = 100.0 * sum(1 for r in rows if r.get("detected")) / len(rows) if rows else 0.0
    print(f"Frames: {len(rows)}, detected: {detected_rate:.1f}%")

    valid = [
        r
        for r in rows
        if r.get("detected")
        and np.isfinite(r.get("pool_x_m", math.nan))
        and np.isfinite(r.get("pool_y_m", math.nan))
    ]
    if valid:
        xs = [float(r["pool_x_m"]) for r in valid]
        ys = [float(r["pool_y_m"]) for r in valid]
        print(
            "Metric coordinate range: "
            f"x={min(xs):.3f}..{max(xs):.3f} m, "
            f"y={min(ys):.3f}..{max(ys):.3f} m"
        )


if __name__ == "__main__":
    main()
