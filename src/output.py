import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import TrackerConfig, target_thruster_count
from tracker_types import Point, ThrusterPoints


def compute_speeds(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Populate `speed_mps` from consecutive detected metric positions."""
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


def write_csv(rows: List[Dict[str, Any]], path: str, num_thrusters: int) -> None:
    """Write tracking rows to CSV with dynamic thruster columns."""
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
        "thruster_min_distance_px",
        "thruster_max_distance_px",
    ]
    for idx in range(1, num_thrusters + 1):
        fieldnames.extend([f"thruster_{idx}_x", f"thruster_{idx}_y"])
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, math.nan) for key in fieldnames})


def draw_annotation(
    frame: np.ndarray,
    detections: List[Dict[str, float]],
    thruster_points: ThrusterPoints,
    roi_centers: Optional[ThrusterPoints],
    tracking_mode: str,
    pos_px: Optional[Point],
    pool_xy: Tuple[float, float],
    trail: List[Point],
    cfg: TrackerConfig,
    frame_idx: int,
    time_s: float,
    detected: bool,
) -> np.ndarray:
    """Render detections, ROI hints, tracking mode, and trail onto one frame."""
    out = frame.copy()
    n_thrusters = target_thruster_count(cfg)

    def draw_polygon(corners: List[Point], color: Tuple[int, int, int], label: str) -> None:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = int(pts[0][0]), int(pts[0][1])
        # 左上点の近くにラベルを出し、2つの四角形が重なっても用途を判別できるようにする。
        cv2.putText(out, label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if cfg.pool_corners_px is not None:
        draw_polygon(cfg.pool_corners_px, (255, 255, 0), "pool")
    if cfg.water_area_corners_px is not None:
        draw_polygon(cfg.water_area_corners_px, (255, 0, 255), "water")

    for detection in detections[:20]:
        x, y, w, h = int(detection["x"]), int(detection["y"]), int(detection["w"]), int(detection["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 128, 255), 1)

    if roi_centers is not None:
        search_radius = max(1, int(round(float(cfg.thruster_search_radius_px))))
        reacquire_radius = max(search_radius, int(round(float(cfg.thruster_reacquire_radius_px))))
        for idx, center in enumerate(roi_centers, start=1):
            x, y = int(round(center[0])), int(round(center[1]))
            cv2.circle(out, (x, y), reacquire_radius, (255, 255, 0), 1)
            cv2.circle(out, (x, y), search_radius, (255, 0, 0), 1)
            cv2.drawMarker(out, (x, y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
            cv2.putText(out, f"R{idx}", (x + 5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f"R{idx}", (x + 5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    if len(trail) >= 2:
        pts = np.array([(int(x), int(y)) for x, y in trail[-200:]], dtype=np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 0), 2)

    if tracking_mode == "roi":
        thruster_color = (0, 165, 255)
    elif tracking_mode == "hold":
        thruster_color = (0, 255, 255)
    else:
        thruster_color = (0, 0, 255)
    for idx, point in enumerate(thruster_points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(out, (x, y), 7, thruster_color, 2)
        cv2.putText(out, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, thruster_color, 1, cv2.LINE_AA)

    if pos_px is not None:
        x, y = int(pos_px[0]), int(pos_px[1])
        marker_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.circle(out, (x, y), 12, marker_color, 2)
        cv2.drawMarker(out, (x, y), marker_color, cv2.MARKER_CROSS, 24, 2)

    label1 = f"frame={frame_idx}  t={time_s:.3f}s  detected={detected}"
    if np.isfinite(pool_xy[0]) and np.isfinite(pool_xy[1]):
        label2 = f"x={pool_xy[0]:.3f} m, y={pool_xy[1]:.3f} m"
    elif pos_px is not None:
        label2 = f"px=({pos_px[0]:.1f}, {pos_px[1]:.1f})"
    else:
        label2 = "position=NaN"
    label3 = f"thrusters={len(thruster_points)}/{n_thrusters}"
    label4 = f"tracking={tracking_mode}"
    for idx, text in enumerate([label1, label2, label3, label4]):
        y = 35 + idx * 32
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out
