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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class TrackerConfig:
    # HSV threshold for orange objects. OpenCV hue range is 0..179.
    hsv_lower: Tuple[int, int, int] = (5, 80, 50)
    hsv_upper: Tuple[int, int, int] = (30, 255, 255)

    # Contour filters.
    min_area_px: float = 20.0
    max_area_px: float = 30000.0

    # Tracking parameters.
    # If the robot has multiple orange thrusters, contours inside cluster_radius_px
    # around the chosen contour are averaged as one robot position.
    cluster_radius_px: float = 80.0
    max_jump_px: float = 180.0
    smoothing_alpha: float = 0.35  # 0=no update, 1=no smoothing

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


def detect_orange_contours(frame: np.ndarray, cfg: TrackerConfig, pool_mask: Optional[np.ndarray]) -> Tuple[List[Dict[str, float]], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if pool_mask is not None:
        mask = cv2.bitwise_and(mask, pool_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
    return detections, mask


def dist(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def choose_robot_position(
    detections: List[Dict[str, float]],
    cfg: TrackerConfig,
    prev_pos: Optional[Point],
    velocity: Point,
) -> Tuple[Optional[Point], float, int, bool]:
    """
    Selects the orange contour likely to be the robot, then averages nearby contours.
    Returns: position_px, total_area, n_cluster_contours, detected_reliably
    """
    if not detections:
        return None, 0.0, 0, False

    if prev_pos is None:
        if cfg.init_point_px is not None:
            target = cfg.init_point_px
            chosen = min(detections, key=lambda d: dist((d["cx"], d["cy"]), target))
        else:
            chosen = detections[0]
        reliable = True
    else:
        predicted = (prev_pos[0] + velocity[0], prev_pos[1] + velocity[1])
        # Prefer candidates close to prediction. Area is a weak tie-breaker.
        scored = []
        for d in detections:
            p = (d["cx"], d["cy"])
            jump = dist(p, predicted)
            score = jump - 0.015 * math.sqrt(max(d["area"], 0.0))
            scored.append((score, jump, d))
        scored.sort(key=lambda t: t[0])
        _, jump, chosen = scored[0]
        reliable = jump <= cfg.max_jump_px
        if not reliable:
            return None, 0.0, 0, False

    cp = (chosen["cx"], chosen["cy"])
    cluster = [d for d in detections if dist((d["cx"], d["cy"]), cp) <= cfg.cluster_radius_px]
    if not cluster:
        cluster = [chosen]
    total_area = sum(max(d["area"], 1.0) for d in cluster)
    cx = sum(d["cx"] * max(d["area"], 1.0) for d in cluster) / total_area
    cy = sum(d["cy"] * max(d["area"], 1.0) for d in cluster) / total_area
    return (float(cx), float(cy)), float(total_area), len(cluster), reliable


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
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, math.nan) for k in fieldnames})


def draw_annotation(
    frame: np.ndarray,
    detections: List[Dict[str, float]],
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
    for i, text in enumerate([label1, label2]):
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if args.annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotated, fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write annotated video: {args.annotated}")

    rows: List[Dict[str, Any]] = []
    prev_pos: Optional[Point] = cfg.init_point_px
    smoothed_pos: Optional[Point] = None
    velocity: Point = (0.0, 0.0)
    trail: List[Point] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps if fps > 0 else float(frame_idx)
        detections, mask = detect_orange_contours(frame, cfg, pool_mask)

        raw_pos, area, n_cluster, reliable = choose_robot_position(detections, cfg, prev_pos, velocity)
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
            pos_for_output: Optional[Point] = smoothed_pos
            trail.append(smoothed_pos)
        else:
            pos_for_output = None

        pool_xy = transform_point(H, pos_for_output)
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
            }
        )

        if writer is not None:
            annotated = draw_annotation(frame, detections, pos_for_output, pool_xy, trail, cfg, frame_idx, time_s, detected)
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
